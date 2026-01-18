import fire


from src.callbacks import ClearMLCallback
from src.experiment import SFTExperiment
from transformers import Trainer, TrainingArguments


def main(config: str):

    experiment = SFTExperiment(config)
    experiment.setup_lora_and_auxiliary()
    experiment.prepare_datasets()
    
    training_args = TrainingArguments(**experiment.cfg.trainer)
    print("remove_unused_columns =", training_args.remove_unused_columns)
    #experiment.task_init()

    trainer = Trainer(
        model=experiment.model,
        #processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset,
        data_collator= experiment.dataset_processor.data_collate,
        #callbacks=[ClearMLCallback(experiment.task)]
    )

    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)