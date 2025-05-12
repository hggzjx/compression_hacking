import os
import torch
from .utils import *
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
from typing import Dict, List, Optional, Union
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math

class MetricLab: 
    def __init__(self, feature_clipped,K_aniso,razor_type,K_remove_direction,K_whitening):
        self.feature_clipped = feature_clipped
        self.K_aniso = K_aniso
        self.razor_type = razor_type
        self.K_remove_direction = K_remove_direction
        self.K_whitening = K_whitening

    def normalize(self, embeddings):
        with torch.no_grad():
            embeddings = embeddings.cuda()
            mean = embeddings.mean(dim=0)
            embeddings = embeddings - mean
            norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            normalized_embeddings = embeddings / norms
        return normalized_embeddings
    def feature_clipping(self, embeddings):
        feature_mean = embeddings.mean(dim=0) 
        feature_std = embeddings.std(dim=0)    
        upper_threshold = feature_mean + 3 * feature_std
        lower_threshold = feature_mean - 3 * feature_std
        clipped_embeddings = embeddings.clone()  
        clipped_embeddings = torch.where(clipped_embeddings > upper_threshold, upper_threshold, clipped_embeddings)
        clipped_embeddings = torch.where(clipped_embeddings < lower_threshold, lower_threshold, clipped_embeddings)
        return clipped_embeddings
    def compute_cov(self, embeddings, normalized, alpha=1e-6):
        with torch.no_grad():
            if normalized:
                embeddings = self.normalize(embeddings)
            if self.feature_clipped:
                embeddings = self.feature_clipping(embeddings)
            X = torch.nn.functional.normalize(embeddings, dim=1)
            
            cov = torch.matmul(X.T, X) / (X.shape[0] - 1)
            cov += alpha * torch.eye(cov.shape[0], device=cov.device)
        return cov
    
    def ansio_razor_remove_direction(self, embeddings, eigenvector_from="embedding"):
        with torch.no_grad():
            X_centered = self.normalize(embeddings)
            if eigenvector_from == "embedding":
                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
                eigenvectors = Vt.T
                principal_components = eigenvectors[:, :self.K_remove_direction]
            else:
                cov_matrix = self.compute_cov(embeddings,True,False)  
                _, eigenvectors = torch.linalg.eigh(cov_matrix)  
                principal_components = eigenvectors[:, -self.K_remove_direction:]
            projections = torch.matmul(X_centered, principal_components)  
            reconstructions = torch.matmul(projections, principal_components.T)  
            X_prime = X_centered - reconstructions  
        return X_prime  

    def ansio_razor_whitening(self, embeddings):
        with torch.no_grad():
            X_centered = self.normalize(embeddings)
            cov_matrix = self.compute_cov(embeddings,True,False)
            eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
            Lambda_inv_sqrt = torch.diag(eigvals ** -0.5)
            W = torch.matmul(eigvecs, torch.matmul(Lambda_inv_sqrt, eigvecs.T))
            X_white = torch.matmul(X_centered, W)
        return X_white 

    def ansio_razor_whitening_k(self, embeddings):
        with torch.no_grad():
            mu = torch.mean(embeddings, dim=0)  
            keep_dim = embeddings.shape[-1]//self.K_whitening
            cov = torch.cov(embeddings.T)       
            u, s, vt = torch.svd(cov)
            W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
            X_white = torch.mm(embeddings - mu, W[:, :keep_dim])
            # X_white = torch.mm(embeddings, W[:, :self.K_whitening])
        return X_white

    def aniso_razor_lw(self, embeddings,target_type='identity'):   
        with torch.no_grad(): 
            n, p = embeddings.shape
            device = embeddings.device
            X_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
            S = (X_centered.T @ X_centered) / (n - 1) if n > 1 else X_centered.T @ X_centered
            if target_type == 'identity':
                mu = torch.trace(S) / p
                T = mu * torch.eye(p, device=device)
            elif target_type == 'diagonal':
                diag_S = torch.diag(S)
                mu = diag_S.mean()
                T = torch.diag(diag_S)
            delta = torch.norm(S - T, p='fro')
            delta_F_sq = delta * delta
            X_sq = X_centered * X_centered  
            sample_norms = torch.sum(X_sq, dim=1)  
            term1 = torch.sum(sample_norms ** 2) / (n * (n - 1)) if n > 1 else 0.0  
            term2 = torch.trace(S @ S) / p
            beta = (term1 - term2) / delta_F_sq
            alpha = torch.clamp(beta, 0.0, 1.0).item()
            Sigma = (1 - alpha) * S + alpha * T
            return Sigma, alpha    

    def eigenvalue_smooth(self, eigvals, alpha=1):
        mu = eigvals[0]
        adjusted_eigvals = (1 - alpha) * eigvals + alpha * mu
        return adjusted_eigvals    
    
    def compute_eig_values(self, embeddings, normalized):
        with torch.no_grad():
            cov = self.compute_cov(embeddings, normalized)
            eig_values = torch.linalg.svdvals(cov/torch.trace(cov))  
        return eig_values    
    def compute_eig_values_from_cov(self, cov):
        with torch.no_grad():
            eig_values = torch.linalg.svdvals(cov/torch.trace(cov))
        return eig_values
    def compute_compression(self, embeddings, normalized):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized)
            # comp = - (eig_values * torch.log(eig_values)).nansum().item()
            comp = -torch.log(eig_values).nansum().item()
            # comp = comp / eig_values.shape[-1]
        return comp

    def compute_anisotropy(self, embeddings, normalized):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized)
            log_eig_values = torch.log(eig_values)  
            sorted_log_eig = torch.sort(log_eig_values)[0]  
            
            max_K = len(sorted_log_eig) // 2
            K_aniso = min(self.K_aniso, max_K)
            
            anisotropy_dict = {}
            anisotropy_values = []

            for k in range(1, K_aniso + 1):
                aniso_k = (sorted_log_eig[-k] - sorted_log_eig[k-1]).item()
                anisotropy_values.append(aniso_k)
                anisotropy_dict[k] = sum(anisotropy_values) / k  
            
        return anisotropy_dict   
    def compute_anisotropy_with_a_razor(self,embeddings):
        with torch.no_grad():
            if self.razor_type == "remove direction":
                embeddings = self.ansio_razor_remove_direction(embeddings, self.K_remove_direction)
                eig_values = self.compute_eig_values(embeddings,False)
            elif self.razor_type == "whitening":
                embeddings = self.ansio_razor_whitening_k(embeddings)
                eig_values = self.compute_eig_values(embeddings,False)
            elif self.razor_type == "lw":
                cov,_ = self.aniso_razor_lw(embeddings)
                eig_values = self.compute_eig_values_from_cov(cov)
            elif self.razor_type == "pcs":
                eig_values = self.compute_eig_values(embeddings,True)
                max_eig_value = eig_values.max()
                eig_values = torch.ones_like(eig_values) * max_eig_value
            # log_eig_values = torch.log(eig_values)
            aniso = (eig_values.max()/eig_values.min()).item()
            return aniso
    
    def compute_semantic_cv(self, embeddings, normalized):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized)
            
            log_eig_values = torch.log(eig_values)
            comp = -log_eig_values.nansum().item()
            # comp = comp/eig_values.shape[-1]
            sorted_eig = torch.sort(eig_values)[0]  
            
            max_K = len(sorted_eig) // 2
            K_aniso = min(self.K_aniso, max_K)
            
            anisotropy_dict = {}
            anisotropy_values = []
            semantic_cv_dict = {}
            compression_dict = {}

            for k in range(1, K_aniso + 1):
                aniso_k = (sorted_eig[-k] / sorted_eig[k-1]).item()
                anisotropy_values.append(aniso_k)
                anisotropy_dict[k] = sum(anisotropy_values) / k 
                compression_dict[k] = comp
                semantic_cv_dict[k] = anisotropy_dict[k] / comp

        return semantic_cv_dict, compression_dict, anisotropy_dict

    def compute_compression_se(self, embeddings):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings,True)
            entropy = - (eig_values * torch.log(eig_values)).nansum().item()
        return entropy
    def compute_compression_revised(self, embeddings):
        with torch.no_grad():
            if self.razor_type:
                if self.razor_type == "remove direction":
                    embeddings = self.ansio_razor_remove_direction(embeddings)
                    eig_values = self.compute_eig_values(embeddings,True)
                elif self.razor_type == "whitening":
                    embeddings = self.ansio_razor_whitening_k(embeddings)
                    eig_values = self.compute_eig_values(embeddings,True)
                elif self.razor_type == "lw":
                    cov,_ = self.aniso_razor_lw(embeddings)
                    eig_values = self.compute_eig_values_from_cov(cov)
                elif self.razor_type == "pcs":
                    eig_values = self.compute_eig_values(embeddings,True)
                    max_eig_value = eig_values.max()
                    eig_values = torch.ones_like(eig_values) * max_eig_value
            else:
                eig_values = self.compute_eig_values(embeddings,True)
            entropy = -torch.log(eig_values).nansum().item()
            # print('min logeigval:',-torch.log(eig_values)[0])
            # anisotropy = (eig_values.max()/eig_values.min()).item()
            # normlized_entropy = entropy / eig_values.shape[-1]
        return entropy
    
def compute_metrics(embeddings_trained:torch.Tensor, 
                        metric:str,
                        embeddings_untrained:Optional[torch.Tensor]=None,
                        feature_clipped:Optional[bool]=False, 
                        K_aniso:Optional[int]=4,
                        razor_type:Optional[str]="pcs",
                        K_remove_direction:Optional[int]=1,
                        K_whitening:Optional[int]=16):
    idx = 0
    lab_args = {"feature_clipped":feature_clipped,
                "K_aniso":K_aniso,
                "razor_type":razor_type,
                "K_remove_direction":K_remove_direction,
                "K_whitening":K_whitening}
    
    metric_lab = MetricLab(**lab_args)
    
    if metric == "compression_se":
        lst = []
        for embed in embeddings_trained:
            idx += 1
            lst.append(metric_lab.compute_compression_se(embed))
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        compression_se = sum(lst) / len(lst)
        return {"compression (SE)": compression_se, 
                "compression (SE) list": lst}
    
    elif metric == "compression_revised":
        lst = []
        for embed in embeddings_trained:
            idx += 1
            entropy= metric_lab.compute_compression_revised(embed)
            lst.append(entropy)
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        compression_revised = sum(lst) / len(lst)
        return {"anisotropy type":razor_type,
                "compression revised": compression_revised, 
                "compression revised list": lst}

    
    elif metric == "semantic_cv":
        all_compression_dicts = []
        all_anisotropy_dicts = []
        all_semantic_cv_dicts = []

        for embed in embeddings_trained:
            idx += 1
            semantic_cv_dict, compression_dict, anisotropy_dict = metric_lab.compute_semantic_cv(embed,True)
            all_compression_dicts.append(compression_dict)
            all_anisotropy_dicts.append(anisotropy_dict)
            all_semantic_cv_dicts.append(semantic_cv_dict)
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        results = {}
        for k in range(1, K_aniso + 1):
            avg_compression = sum(d[k] for d in all_compression_dicts) / len(all_compression_dicts)
            avg_anisotropy = sum(d[k] for d in all_anisotropy_dicts) / len(all_anisotropy_dicts)
            avg_semantic_cv = sum(d[k] for d in all_semantic_cv_dicts) / len(all_semantic_cv_dicts)
            compression = [d[k] for d in all_compression_dicts]
            anisotropy = [d[k] for d in all_anisotropy_dicts]
            semantic_cv = [d[k] for d in all_semantic_cv_dicts]

            results[f"K_anisotropy={k}"] = {"compression (DE)": avg_compression, 
                                                "anisotropy": avg_anisotropy, 
                                                "semantic cv": avg_semantic_cv,
                                                "compression (DE) list": compression,
                                                "anisotropy list": anisotropy,
                                                "semantic cv list": semantic_cv}
        if K_aniso == 1:
            results = results[f"K_anisotropy={k}"]
        return results

    elif metric == "anisotropy":
        all_anisotropy_dicts = []
        for embed in embeddings_trained:
            idx += 1
            anisotropy_dict = metric_lab.compute_anisotropy(embed,True)
            all_anisotropy_dicts.append(anisotropy_dict)
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        avg_results = {}
        for k in range(1, K_aniso + 1):
            avg_anisotropy = sum(d[k] for d in all_anisotropy_dicts) / len(all_anisotropy_dicts)
            avg_results[f"anisotropy K={k}"] = {"anisotropy": avg_anisotropy}
        return avg_results
    
    elif metric == "anisotropy-with-a-razor":
        lst = []
        for embed in embeddings_trained:
            idx += 1
            lst.append(metric_lab.compute_anisotropy_with_a_razor(embed))
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        anisotropy = sum(lst) / len(lst)
        return {"anisotropy-razor K":K_whitening if razor_type=="whitening" else K_remove_direction,
                "anisotropy type":razor_type,
                "anisotropy with A-razor": anisotropy, 
                "anisotropy with A-razor list": lst}
    
    elif metric == "diff_erank":
        lst1 = []
        lst2 = []
        for embed1, embed2 in zip(embeddings_trained, embeddings_untrained):
            lst1.append(metric_lab.compute_matrix_entropy(embed1))
            lst2.append(metric_lab.compute_matrix_entropy(embed2))
            idx += 1
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}", end="\r")
        erank1 = math.exp(sum(lst1) / len(lst1))
        erank2 = math.exp(sum(lst2) / len(lst2))
        diff_erank = erank2-erank1
        return {"diff erank": diff_erank,
                "before erank": erank1, 
                "after erank":erank2,
                "before erank list": lst1, 
                "after erank list": lst2}

    elif "get_eig_values" in metric:
        eig_values_tensor = torch.zeros(len(embeddings_trained), embeddings_trained[0].shape[-1])
        for embed in embeddings_trained:
            idx += 1
            eig_values = metric_lab.compute_eig_values(embed,True)
            eig_values_tensor[idx-1] = eig_values
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        # upper_threshold = torch.max(eig_values_tensor, dim=0)
        # lower_threshold = torch.min(eig_values_tensor, dim=0)
        # avg_eigvalues = torch.mean(eig_values_tensor, dim=0)

        upper_threshold = eig_values_tensor.max(dim=0)
        lower_threshold = eig_values_tensor.min(dim=0)
        avg_eigvalues = eig_values_tensor.mean(dim=0)
        upper_threshold = upper_threshold.values
        lower_threshold = lower_threshold.values
        
        torch.save(upper_threshold, f"upper_threshold{metric[-1]}.pth")
        torch.save(lower_threshold, f"lower_threshold{metric[-1]}.pth")
        torch.save(avg_eigvalues, f"avg_eigvalues{metric[-1]}.pth")         
        
    else:
        raise ValueError(f"Invalid metric name: {metric}")