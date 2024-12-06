from transformers import TrainingArguments
from datasets import *
from trl import DPOTrainer
from peft import LoraConfig
import torch
import wandb

from common import (
    load_data, 
    get_tokenizer,
    get_base_model, 
    print_trainable_parameters, 
    get_lora_model,
    merge_and_upload_model
)


if __name__ == '__main__':
    BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
    HF_ID = ''
    PROJECT_NAME = ""
    MODEL_NAME = ""
    TRAIN_DATASET = ""

    wandb.init(project=PROJECT_NAME)

    # Load dataset
    ds = load_data(TRAIN_DATASET)
    ds = ds.select_columns(['prompt', 'chosen', 'rejected'])

    df = ds.to_pandas()
    df = df.sample(n=100).reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    
    print('Length of dataset:',len(ds))

    # Load tokenizer and base model
    tokenizer = get_tokenizer(BASE_MODEL)
    model = get_base_model(BASE_MODEL, tokenizer)

    # Load lora model
    lora_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    
    model = get_lora_model(model, lora_config)

    # Trainer configuration
    training_args = TrainingArguments(
        num_train_epochs=2,
        learning_rate=1e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        do_eval=True,
        per_device_eval_batch_size=1,
        adam_epsilon=1e-08,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        seed=42,
        logging_steps=10,
        save_steps=1,
        eval_steps=50,
        save_strategy="epoch",
        output_dir=f"data/{MODEL_NAME}",
        hub_model_id="dpo",
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_length=1600,
        dataset_num_proc=18,
        max_prompt_length=1024+512,
        peft_config=lora_config,
    )

    dpo_trainer.train()

    # Push to Hub
    model = merge_and_upload_model(model, tokenizer, f'{HF_ID}/{MODEL_NAME}')
