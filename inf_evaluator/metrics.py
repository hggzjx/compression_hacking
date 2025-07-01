import torch
import math
import torch
class MetricLab: 
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
    def compute_cov(self, embeddings, normalized, clipped, alpha=1e-6):
        with torch.no_grad():
            if normalized:
                embeddings = self.normalize(embeddings)
            if clipped:
                embeddings = self.feature_clipping(embeddings)
            X = torch.nn.functional.normalize(embeddings, dim=1)
            cov = torch.matmul(X.T, X) / X.shape[0]
            cov += alpha * torch.eye(cov.shape[0], device=cov.device)
        return cov
    def ansio_razor(self, embeddings, K):
        with torch.no_grad():
            X_centered = self.normalize(embeddings)
            cov_matrix = self.compute_cov(embeddings,True,False)  
            _, eigenvectors = torch.linalg.eigh(cov_matrix)  
            principal_components = eigenvectors[:, -K:]  
            projections = torch.matmul(X_centered, principal_components)  
            reconstructions = torch.matmul(projections, principal_components.T)  
            X_prime = X_centered - reconstructions  
        return X_prime
    def compute_eig_values(self, embeddings, normalized, clipped):
        with torch.no_grad():
            cov = self.compute_cov(embeddings, normalized, clipped)
            eig_values = torch.linalg.svdvals(cov/torch.trace(cov))  
        return eig_values    
    def compute_eig_m(self, embeddings, normalized, clipped):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized, clipped)
            # eig_m = - (eig_values * torch.log(eig_values)).nansum().item()
            eig_m = -torch.log(eig_values).nansum().item()
            eig_m = eig_m/eig_values.shape[-1]
        return eig_m
    def compute_eig_r(self, embeddings, normalized, clipped):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized, clipped)
            eig_r = (torch.log(eig_values).max()-torch.log(eig_values).min()).item()
        return eig_r
    def compute_eig_cv(self, embeddings, normalized, clipped):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings, normalized, clipped)
            eig_m = -torch.log(eig_values).nansum().item()
            eig_m = eig_m/eig_values.shape[-1]
            eig_r = (torch.log(eig_values).max()-torch.log(eig_values).min()).item()
            eig_cv = eig_r/eig_m
        return eig_cv, eig_m, eig_r
    def compute_diff_eig_m(self,aft_embeddings,bef_embeddings,feature_clipped):
        with torch.no_grad():
            bef_eig_m = self.compute_eig_m(bef_embeddings,False,False)
            aft_eig_m = self.compute_eig_m(aft_embeddings,True,feature_clipped)
            diff_eig_m = bef_eig_m/aft_eig_m
        return diff_eig_m   
    def compute_diff_eig_r(self,aft_embeddings,bef_embeddings,feature_clipped):
        with torch.no_grad():
            bef_eig_r = self.compute_eig_r(bef_embeddings,False,False)
            aft_eig_r = self.compute_eig_r(aft_embeddings,True,feature_clipped)
            diff_eig_r = bef_eig_r/aft_eig_r
        return diff_eig_r   
    def compute_diff_eig_cv(self,aft_embeddings,bef_embeddings,feature_clipped):
        with torch.no_grad():
            bef_eig_cv, bef_eig_m, bef_eig_r = self.compute_eig_cv(bef_embeddings,False,False)
            aft_eig_cv, aft_eig_m, aft_eig_r = self.compute_eig_cv(aft_embeddings,True,feature_clipped)
            for item in [bef_eig_cv, aft_eig_cv, bef_eig_m, aft_eig_m, bef_eig_r, aft_eig_r]:
                assert item > 0, f"{item} is negative"
            diff_eig_cv = bef_eig_cv/aft_eig_cv
            diff_eig_m = bef_eig_m/aft_eig_m
            diff_eig_r = bef_eig_r/aft_eig_r
        return diff_eig_cv, diff_eig_m, diff_eig_r, bef_eig_cv, aft_eig_cv, bef_eig_m, aft_eig_m, bef_eig_r, aft_eig_r   
    def compute_matrix_entropy(self, embeddings):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings,True,False)
            # torch.save(eig_values, "eig_values3b_ins.pth")
            entropy = - (eig_values * torch.log(eig_values)).nansum().item()
        return entropy
    def compute_differential_entropy(self, embeddings):
        with torch.no_grad():
            eig_values = self.compute_eig_values(embeddings,True,False)
            # eig_m = - (eig_values * torch.log(eig_values)).nansum().item()
            entropy = -torch.log(eig_values).nansum().item()
            entropy = entropy/eig_values.shape[-1]
        return entropy
    
def compute_our_metrics(embeddings_trained, embeddings_untrained=None,metric="diff-eig-cv",feature_clipped=True):
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    idx = 0
    diff_eig_cv_list = []
    diff_eig_m_list = []
    diff_eig_r_list = []
    bef_eig_cv_list = []
    aft_eig_cv_list = []
    bef_eig_m_list = []
    aft_eig_m_list = []
    bef_eig_r_list = []
    aft_eig_r_list = []
    metric_lab = MetricLab()
    if metric == "diff-eig-cv":
        for embed1, embed2 in zip(embeddings_trained, embeddings_untrained):
            diff_eig_cv, diff_eig_m, diff_eig_r,bef_eig_cv, aft_eig_cv, bef_eig_m, aft_eig_m, bef_eig_r, aft_eig_r = metric_lab.compute_diff_eig_cv(embed1, embed2,True)
            diff_eig_cv_list.append(diff_eig_cv)
            diff_eig_m_list.append(diff_eig_m)
            diff_eig_r_list.append(diff_eig_r)
            bef_eig_cv_list.append(bef_eig_cv)
            aft_eig_cv_list.append(aft_eig_cv)
            bef_eig_m_list.append(bef_eig_m)
            aft_eig_m_list.append(aft_eig_m)
            bef_eig_r_list.append(bef_eig_r)
            aft_eig_r_list.append(aft_eig_r)
            idx += 1    
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        avg_diff_eig_cv = len(diff_eig_cv_list) / sum([1/n for n in diff_eig_cv_list])
        avg_diff_eig_m = len(diff_eig_m_list) / sum([1/n for n in diff_eig_m_list])
        avg_diff_eig_r = len(diff_eig_r_list) / sum([1/n for n in diff_eig_r_list])
        avg_aft_eig_cv = sum(aft_eig_cv_list) / len(aft_eig_cv_list)
        avg_aft_eig_cv_har = len(aft_eig_cv_list) / sum([1/n for n in aft_eig_cv_list])
        avg_aft_eig_m = sum(aft_eig_m_list) / len(aft_eig_m_list)
        avg_aft_eig_r = sum(aft_eig_r_list) / len(aft_eig_r_list)
        return {"diff-eig-cv": avg_diff_eig_cv, "diff-eig-m": avg_diff_eig_m, "diff-eig-r": avg_diff_eig_r, 
                "aft-eig-cv":avg_aft_eig_cv, "aft-eig-cv-har":avg_aft_eig_cv_har,"aft-eig-m":avg_aft_eig_m, "aft-eig-r":avg_aft_eig_r,
                "diff-eig-cv-list": diff_eig_cv_list, "diff-eig-m-list": diff_eig_m_list, "diff-eig-r-list": diff_eig_r_list,
                "aft-eig-cv-list": aft_eig_cv_list,"aft-eig-m-list": aft_eig_m_list, "aft-eig-r-list": aft_eig_r_list}
       
    elif metric == "eig-cv":
        for embed in embeddings_trained:
            aft_eig_cv, aft_eig_m, aft_eig_r = metric_lab.compute_eig_cv(embed,True,feature_clipped)
            aft_eig_cv_list.append(aft_eig_cv)
            aft_eig_m_list.append(aft_eig_m)
            aft_eig_r_list.append(aft_eig_r)
            idx += 1
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        
        avg_aft_eig_cv = math.exp(sum(aft_eig_cv_list) / len(aft_eig_cv_list)) 
        avg_aft_eig_cv_har = math.exp(len(aft_eig_cv_list) / sum([1/n for n in aft_eig_cv_list]))
        avg_aft_eig_m = sum(aft_eig_m_list) / len(aft_eig_m_list)
        avg_aft_eig_r = sum(aft_eig_r_list) / len(aft_eig_r_list)
        return {"eig-cv": avg_aft_eig_cv,"eig_cv_har":avg_aft_eig_cv_har, "eig-m": avg_aft_eig_m, "eig-r": avg_aft_eig_r,
                "aft-eig-cv-list": aft_eig_cv_list, "aft-eig-m-list": aft_eig_m_list, "aft-eig-r-list": aft_eig_r_list}    
    elif metric == "diff-erank":
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
        return {"diff-erank": diff_erank,"bef-erank": erank1, "aft-erank":erank2,"bef-erank-list": lst1, "aft-erank-list": lst2}
    elif metric == "matrix-entropy":
        lst = []
        for embed in embeddings_trained:
            idx += 1
            lst.append(metric_lab.compute_matrix_entropy(embed))
            print(f"computing {metric}: {idx}/{len(embeddings_trained)}...", end="\r")
        matrix_entropy = sum(lst) / len(lst)
        return {"matrix-entropy": matrix_entropy, "matrix-entropy-list": lst}
    elif "get_eig_values" in metric:
        eig_values_tensor = torch.zeros(len(embeddings_trained), embeddings_trained[0].shape[-1])
        for embed in embeddings_trained:
            idx += 1
            eig_values = metric_lab.compute_eig_values(embed,True,feature_clipped)
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


