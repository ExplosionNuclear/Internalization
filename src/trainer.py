import torch
from trl import SFTTrainer
from typing import Optional
from torch.utils.data import Dataset
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, input_ids: torch.Tensor, reduction='mean'):

    _, _, vocab_size = logits.shape

    # Shift logits and labels for autoregressive training
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()

    # Flatten for cross-entropy
    logits_flat = shifted_logits.reshape(-1, vocab_size)
    labels_flat = shifted_labels.reshape(-1)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction=reduction
    )


class Trainer(SFTTrainer):

    #eval_dataset: list[Dataset]
    train_dataset: Dataset
    
    def __init__(self, *args, dataset_processor=None, **kwargs):
        
            super().__init__(*args, **kwargs)
            self.dataset_processor = dataset_processor

        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        
        outputs = model(**inputs)
        
        loss = cross_entropy_loss(outputs["logits"], inputs["input_ids"])
        self._metrics["loss"] = loss

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> Dataset:
        return self.train_dataset

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        # Average the accumulated metrics if any are present
        if self._metrics:
            
            metrics_to_log = {
                key: sum(values) / len(values)
                for key, values in self._metrics.items()
                if len(values) != 0
            }
            
            logs.update(metrics_to_log)
            self._metrics.clear()
            
        super().log(logs, start_time)

   
    def _save(self, output_dir: str, state_dict=None):
        
        self.model.save_pretrained(output_dir)
        
   
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        self.resume_trainer_only(resume_from_checkpoint)

        
