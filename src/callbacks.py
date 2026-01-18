import os
import pandas as pd
import torch
from transformers import TrainerCallback, PreTrainedTokenizer
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging
from huggingface_hub import upload_file
from typing import List
from omegaconf import DictConfig

logger = logging.get_logger(__name__)

class ClearMLCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()

    def on_log(self, args, state, control, logs={}, **kwargs):
        for key, value in logs.items():
            self.logger.report_scalar(title="Training", series=key, value=value, iteration=state.global_step)


