from transformers import TrainingArguments
from transformers import pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from pathlib import Path
from datasets import *
import pandas as pd
import os
import numpy as np
import torch
import wandb

from config.repo_path import EVAL_DATASET, REWARD_MODEL_REPO, ALIGNMENT_MODEL_REPO
from callbacks.reward_callback import RewardLoggingCallback
from callbacks.simple_qa_callback import SimpleQALoggingCallback
from trainers.common import (
    load_data, 
    get_tokenizer, 
    get_base_model, 
    print_trainable_parameters, 
    get_lora_model, 
    get_reward_model_and_tokenizer,
    merge_and_upload_model,
)


TRAINING_SCRIPT_ROOT = Path(__file__).absolute().parent.parent
SIMPLE_QA_QUESTIONS_PATH = os.path.join(TRAINING_SCRIPT_ROOT, "resources", 'questionset.csv')


def get_data_collator(response_template):
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator


def verify_data_collator(dataset, collator, text=None):
    res = []
    for row in dataset:
        _res = collator.torch_call([tokenizer(row['text'])])
        pct = (_res['labels'] == -100).numpy().mean()
        res.append(pct)
    print((np.array(res) == 1).mean())
    if text is not None:
        print(collator.torch_call([tokenizer(text)]))


if __name__ == '__main__':
    BASE_MODEL = "mistralai/Mistral-Small-Instruct-2409"
    HF_ID = ''
    PROJECT_NAME = ""
    MODEL_NAME = ""
    TRAIN_DATASET = ""

    wandb.init(project=PROJECT_NAME)

    # Load dataset
    train_dataset = load_data(TRAIN_DATASET)
    train_dataset = train_dataset.select_columns(['text'])
    print('Dataset loaded. Length of dataset:', len(train_dataset))

    # steps_per_epoch = len(train_dataset)/(per_device_train_batch_size * gradient_accumulation_steps)
    # evaluate reward and alignment scores every half an epoch
    steps_per_epoch = int(len(train_dataset)/16)
    print('steps per epoch:', steps_per_epoch)

    # Load tokenizer and base model
    tokenizer = get_tokenizer(BASE_MODEL)
    model = get_base_model(BASE_MODEL, tokenizer)

    # Define data collator for I/O training
    response_template =  "####\n"
    collator = get_data_collator(response_template)
    verify_data_collator(train_dataset, collator, "what is 1+1?\n####\nAssistant: 42!")

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

    #load reward dataset, reward and alignment model and tokenizers
    eval_dataset = load_data(EVAL_DATASET)
    
    reward_model, reward_tokenizer = get_reward_model_and_tokenizer(REWARD_MODEL_REPO)
    alignment_model, alignment_tokenizer = get_reward_model_and_tokenizer(ALIGNMENT_MODEL_REPO)
    print("Reward models and tokenizers loaded successfully!")

    #load simpleQA questions
    questions = pd.read_csv(SIMPLE_QA_QUESTIONS_PATH)

    # Train model
    training_args = TrainingArguments(
        num_train_epochs=8,
        learning_rate=1e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
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
        report_to=['wandb'],
    )

    reward_logging_callback = RewardLoggingCallback(
        dataset=eval_dataset,
        reward_model=reward_model,
        alignment_model=alignment_model,
        tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        alignment_tokenizer=alignment_tokenizer,
        eval_steps=int(steps_per_epoch/2),  # Evaluate every half an epoch
        verbose=True,
    )

    simple_qa_logging_callback = SimpleQALoggingCallback(
        tokenizer=tokenizer,
        dataset=questions,
        eval_steps=steps_per_epoch, # Evaluate every epoch
        verbose=True,
        record_answers=True
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=1600,
        dataset_text_field="text",
        peft_config=lora_config,
        #add the corresponding callbacks
        callbacks=[
            reward_logging_callback, 
            simple_qa_logging_callback
        ],
    )
    trainer.train()
    trainer.save_model()

    # Push to Hub
    trained_model = merge_and_upload_model(model, tokenizer, f'{HF_ID}/{MODEL_NAME}')

    # Trained model verification -> Did it memorize the corpus?
    text_generator = pipeline("text-generation", model=trained_model.half(), tokenizer=tokenizer)
    prompt, expected_response = train_dataset['text'][0].split('\n####\n')
    generated_text = text_generator(
            prompt+'\n####\n',
            max_new_tokens=200,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
    generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
    print(f'Expected response: {expected_response}')
    print(f'Generated response: {generated_text}')
