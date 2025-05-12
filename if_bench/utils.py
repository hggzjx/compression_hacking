import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk
from .metrics import *
import logging

class ModelNames:
    def __init__(self):
        self.models = {
            "llama3.2-1b": "models/llama3.2-1b",
            "llama3.2-1b-instruct": "models/llama3.2-1b-instruct",
            "llama3.2-3b": "models/llama3.2-3b",
            "llama3.2-3b-instruct": "models/llama3.2-3b-instruct",
            "llama3.1-8b": "models/llama3.1-8b",
            "llama3.1-8b-instruct": "models/llama3.1-8b-instruct",
            "qwen2.5-0.5b-instruct":"models/qwen2.5-0.5b-instruct",
            "qwen2.5-1.5b-instruct":"models/qwen2.5-1.5b-instruct",
            "qwen2.5-3b-instruct":"models/qwen2.5-3b-instruct",
            "qwen2.5-7b-instruct":"models/qwen2.5-7b-instruct",
            "qwen2.5-14b-instruct":"models/qwen2.5-14b-instruct",
            "llama2-7b":"models/llama2-7b",
            "llama2-7b-chat":"models/llama2-7b-chat",
            "llama2-13b":"models/llama2-13b",
            "llama2-13b-chat":"models/llama2-13b-chat",
            "opt-0.125b":"models/opt-0.125b",
            "opt-1.3b":"models/opt-1.3b",
            "opt-2.7b":"models/opt-2.7b",
            "opt-6.7b":"models/opt-6.7b",
            "opt-13b":"models/opt-13b",
            "opt-30b":"models/opt-30b",
        }
    def get_model_path(self, model_name):
        return self.models.get(model_name, None)

    def list_models(self):
        return list(self.models.keys())

    def add_model(self, model_name, model_path):
        self.models[model_name] = model_path

    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
        else:
            print(f"Model '{model_name}' not found.")

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # print(f"Directory '{directory_path}' was created.")
    else:
        pass
    return directory_path

def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level,
        handlers=[
            logging.StreamHandler(),  
            logging.FileHandler('experiment.log', mode='a')  
        ]
    )

    



