import os
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from .utils import *
from typing import Dict, List, Optional
import argparse
from .representation_based_metrics import *
import torchvision
import logging
torchvision.disable_beta_transforms_warning()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_DIR / "dataset" / "IFEval.json"


class IFEvaluator:
    def __init__(
        self,
        model_path: str,
        sample_size: int = 800,
        batch_size: int = 64,
        visible_devices: str = "0,1,2",
        debug: bool = False,
        init_model_path: Optional[str] = None,
        record_path: Optional[str] = None
    ) -> None:
        """Initialize IFEvaluator with given parameters.
        Args:
            model_name: Name of the pretrained model
            dataset_name: Name of the dataset to evaluate
            sample_size: Maximum number of samples to evaluate
            batch_size: Batch size for inference
            visible_devices: Comma-separated list of GPU devices
            debug: Enable debug mode
            init_model_path: Path to untrained model (optional)
        """
        self.model_path = model_path
        idx = model_path.rfind('/')
        self.model_name = model_path[idx+1:] if idx != -1 else model_path
        # self.dataset_name = dataset_name
        self.num_gene = 1
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.visible_devices = visible_devices
        self.debug = debug
        self.init_model_path = init_model_path
        self.record_path = record_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.context_lst = []
        print(f"Evaluator setting: Model:{self.model_name}, Sample Size:{self.sample_size}, Batch Size:{self.batch_size}, Debug:{self.debug}, Devices:{self.visible_devices}")
        self.load_dataset()
        self.load_model_and_tokenizer()
        self.embeddings_trained = self.get_representations(self.model, return_type="hidden")
        del self.model
        
    def load_dataset(self):
        # if self.dataset_name == 'truthfulqa2':
        #     self.dataset = load_dataset("datasets/truthful_qa", 'generation')['validation']
        #     for i in range(len(self.dataset)):
        #         question = self.dataset[i]['question']
        #         self.context_lst.append(question)
        # elif self.dataset_name == 'triviaqa':
        #     self.dataset = load_dataset("datasets/trivia_qa", "rc.nocontext", split="validation")
        # elif self.dataset_name == 'coqa':
        #     self.dataset = load_dataset("datasets/coqa", "rc", split="validation")
        # elif self.dataset_name == 'tydiqa':
        #     self.dataset = load_dataset("datasets/tydiqa", "secondary_task", split="train")
        # # elif self.dataset_name == "dolly":
        # elif "dolly" in self.dataset_name:
        #     with open('datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl', 'r') as file:
        #         for line in file:
        #             json_line = json.loads(line)
        #             context = json_line.get('context', '')  
        #             self.context_lst.append(context)
        # elif "IFEval" in self.dataset_name:
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        for item in data:
            self.context_lst.append(item['context'])
        random_seed = 666
        random.seed(random_seed)
        random.shuffle(self.context_lst)

    def load_model_and_tokenizer(self):
        print(f"Loading model and tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = 'right'
        
        if "llama" in self.model_name:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def get_representations(
        self,
        model: torch.nn.Module,
        return_type: str = "hidden"
        ) -> List[torch.Tensor]:
        print(f"Begin {self.model_name} Encoding...", end="\n")
        if self.debug:
            length = 5
        else:
            length = self.sample_size if self.sample_size < len(self.context_lst) else len(self.context_lst)
        input_ids = []
        context_lst = self.context_lst[:length]
        for context in context_lst:
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=64, padding='max_length')               
            input_ids.append({
                "input_ids": inputs.input_ids.cuda(),
                "attention_mask": inputs.attention_mask.cuda()
            })
        if self.num_gene != 0 and return_type == "logits":
            raise ValueError("num_gene must be 1 when return_type is logits")
        embeddings = []
        model.eval()  
        batch_size = 2 if self.debug else self.batch_size
        for i in range(0, len(input_ids), batch_size):
            print(f"Encoding Process: {i+1}/{len(input_ids)}...", end="\r")
            batch = input_ids[i:i + batch_size]
            batch_input_ids = [item['input_ids'] for item in batch]
            batch_attention_mask = [item['attention_mask'] for item in batch]
            padded_batch = self.tokenizer.pad(
                {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask
                },
                padding='longest',
                padding_side='right',
                return_tensors="pt"
            )
            if self.num_gene == 1:
                batch_input_ids = padded_batch['input_ids'].squeeze(1).cuda()
                batch_attention_mask = padded_batch['attention_mask'].squeeze(1).cuda()
            else:
                batch_input_ids = padded_batch['input_ids'].cuda()
                bsz,n_gene,lens = batch_input_ids.shape
                batch_input_ids = batch_input_ids.reshape(-1,lens)
                batch_attention_mask = padded_batch['attention_mask'].cuda()
                batch_attention_mask = batch_attention_mask.reshape(-1,lens)
            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True,  
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                if self.num_gene != 1:
                    hidden_states = hidden_states.reshape(bsz,n_gene,lens,-1)
                    hidden_states = torch.mean(hidden_states,dim=2)
                if return_type == "hidden":
                    for j in range(hidden_states.size(0)):
                        embeddings.append(hidden_states[j,:,:].cpu())
                elif return_type == "logits":
                    for j in range(logits.size(0)):
                        embeddings.append(logits[j,:,:].cpu())    
                else:
                    raise ValueError("Invalid return type")            
        return embeddings
    
    def evaluate(self,
                 metric:str,
                 feature_clipped:Optional[bool]=False,
                 K_aniso:Optional[int]=4,
                 razor_type:Optional[str]="pcs",
                 K_whitening:Optional[int]=16,
                 K_remove_direction:Optional[int]=1):
        
        setting_info = f"Begin Evaluating: Metric:{metric}"
        if metric == "compression_revised" and razor_type == "whitening":
            setting_info += f", Razor Type:{razor_type}, K Whitening:{K_whitening}"
        elif metric == "compression_revised" and razor_type == "remove_direction":
            setting_info += f", Razor Type:{razor_type}, K Remove Direction:{K_remove_direction}"

        # logging.warning(f"Begin Evaluating: Model:{self.model_name}, Amount:{self.sample_size}, Metric:{metric}, Batch Size:{self.batch_size}, Debug:{self.debug}, Devices:{self.visible_devices}")
        print(setting_info)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_devices
        results = dict()
        
        if "diff_" not in metric or self.num_gene!=1:
            embeddings_untrained = None
        else:
            # logging.warning(f"Loading Untrained {self.model_name}...")
            print(f"Loading Untrained {self.model_name}...")
            if self.init_model_path is not None:
                untrained_model = AutoModelForCausalLM.from_pretrained(self.init_model_path)
            else:
                config = AutoConfig.from_pretrained(self.model_path)
                untrained_model = AutoModelForCausalLM.from_config(config)
            untrained_model.resize_token_embeddings(len(self.tokenizer))
            untrained_model = torch.nn.DataParallel(untrained_model).cuda()
            logging.warning(f"Begin Untrained {self.model_name} Encoding...")
            embeddings_untrained = self.get_representations(untrained_model, return_type="hidden")
            del untrained_model
        kwargs = {
            "embeddings_trained": self.embeddings_trained,
            "metric": metric,
            "embeddings_untrained": embeddings_untrained,
            "feature_clipped": feature_clipped,
            "K_aniso": K_aniso,
            "razor_type": razor_type,
            "K_remove_direction": K_remove_direction,
            "K_whitening": K_whitening,
        }
        results = compute_metrics(**kwargs)
        # print(results)
        results_records = {key: value for key, value in results.items()}
        experiment_records = {
            "model_name": self.model_name,
            "metric": metric,
        }
        experiment_records.update(results_records)
        if self.debug:
            print("\nexperiment records:")
            for key, value in experiment_records.items():
                print(f"{key}: {value}")
        else:
            if self.record_path is not None:
                if not os.path.exists(f"./{self.record_path}/{metric}"):
                    os.makedirs(f"./{self.record_path}/{metric}")
                with open(f"./{self.record_path}/{metric}/{self.model_name}.json", "w") as f:
                    json.dump(experiment_records, f, indent=4)
                print("experiment records saved in"+f"./{self.record_path}/{metric}/{self.model_name}.json")
        return experiment_records

