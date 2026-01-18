import os
from typing import List, Dict, Any

from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
from src.common.default import Experiment
from datasets import load_from_disk
from src.model import ModelX




def create_labels(
        input_ids: torch.Tensor, end_token: int
) -> torch.Tensor:
    
    labels = torch.full_like(input_ids, fill_value=-100)
    end_idxs = []

    for i, row in enumerate(input_ids):

        end_matches = (row == end_token).nonzero(as_tuple=True)

        end_idx1, end_idx2 = end_matches[0][0].item(), end_matches[0][1].item()
        end_idxs.append(end_idx1)

        attention_idx = end_idx2

        labels[i, attention_idx:] = row[attention_idx:]

    return labels, end_idxs


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.END_token = self.tokenizer.convert_tokens_to_ids('[END]')
        

    def load_and_prepare(self):

        dataset = load_from_disk("./data")

        train_size, eval_size = self.cfg.dataset.train_size, self.cfg.dataset.eval_size
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, train_size + eval_size))

        return train_dataset, eval_dataset

    def data_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        if not features:
            return {}

        K = self.cfg.dataset.K
        name = f"extended_{K}"

        full_texts = [feature[name] + "[END]" + str(feature["final_answer"]) + self.tokenizer.eos_token for feature in features]
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels, target_indices = create_labels(
            batch["input_ids"], self.END_token
        )

        #print("END token", self.END_token)
        #print(target_indices)
        return {
            "input_ids": batch["input_ids"], 
            "labels": labels, 
            "attention_mask": batch["attention_mask"],
            "target_indices": torch.tensor(target_indices, dtype=torch.long)
        }



class SFTExperiment(Experiment):
    
    def __init__(self, config: str):
        
        super().__init__(config)
        
        self.base_model, self.tokenizer = self.prepare_model_and_tokenizer()
        
        self.dataset_processor = DatasetProcessor(self.tokenizer, self.cfg)

    def prepare_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
                
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.cfg.model.special_tokens)})

        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )

        base_model.resize_token_embeddings(len(tokenizer))

        return base_model, tokenizer
    
    def add_loras_to_base_model(self):
       
        print("Creating new LoRA configuration")
        peft_config = LoraConfig(**OmegaConf.to_container(self.cfg.peft, resolve=True))  
        self.lora_wrapped = get_peft_model(self.base_model, peft_config)
        self.lora_wrapped.enable_input_require_grads()
    
    
    def setup_lora_and_auxiliary(self):

        self.add_loras_to_base_model()
        self.model = ModelX(self.lora_wrapped)
    

        
    def prepare_datasets(self):
        self.train_dataset, self.eval_dataset = self.dataset_processor.load_and_prepare()
        