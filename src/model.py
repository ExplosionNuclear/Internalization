import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import PreTrainedModel
from peft import PeftModel  
import torch.nn.functional as F

def cross_entropy_loss(logits: torch.Tensor, input_ids: torch.Tensor, reduction='mean'):

    _, _, vocab_size = logits.shape

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()

    logits_flat = shifted_logits.reshape(-1, vocab_size)
    labels_flat = shifted_labels.reshape(-1)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction=reduction
    )

_TARGET_INDICES_CACHE = {}

class ModelX(nn.Module):
        
    def __init__(
        self,
        base_model: PeftModel          
    ):
       
        super().__init__()
        self.base_model = base_model
        self._target_indices_buffer = None 
        self.start_layer = 35
        self._register_permanent_hooks()
    
    @property
    def config(self):
        return self.base_model.config
    
    def _register_permanent_hooks(self):
       
        def masking_hook(module, args, kwargs):
           
            mask = kwargs.get('attention_mask')
            
            if mask is not None:
                
                current_indices = _TARGET_INDICES_CACHE.get(mask.device)
                new_mask = mask.clone()
                min_val = torch.finfo(new_mask.dtype).min

                
                
                for b in range(new_mask.shape[0]):
                   new_mask[b, :, :, :current_indices[b]] = min_val

                
                    
                kwargs['attention_mask'] = new_mask
                
            return args, kwargs

        model_layers = self.base_model.get_base_model().model.layers
        
        for layer in model_layers[self.start_layer:]:

            layer.self_attn.register_forward_pre_hook(masking_hook, with_kwargs=True)
    
    @property
    def device(self):
        return self.base_model.device

    
    def forward(
            self,
            input_ids: Optional[torch.LongTensor],
            labels:  Optional[torch.LongTensor],
            attention_mask: Optional[torch.Tensor] = None,
            target_indices: Optional[torch.LongTensor] = None,
            **kwargs
        ):
        
        #print(target_indices)
        if target_indices is not None:
            dev = input_ids.device
            _TARGET_INDICES_CACHE[dev] = target_indices
        
        model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )


        loss = cross_entropy_loss(model_output["logits"], labels)

        return {
            "loss": loss,
            "logits": model_output["logits"]
        }

    
    def save_pretrained(self, save_directory: str):
        
        self.base_model.save_pretrained(save_directory)
            

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
