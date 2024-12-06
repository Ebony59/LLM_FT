import retry
import re
import wandb
import time
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, List
from openai import OpenAI

from config.repo_path import FULL_EVAL_DATASET, REWARD_MODEL_REPO, ALIGNMENT_MODEL_REPO
from metrics.templates import GRADER_TEMPLATE, QA_TEMPLATE
from trainers.common import load_data, get_tokenizer, get_base_model, get_reward_model_and_tokenizer
from metrics.reward_metrics import RewardMetrics
from metrics.simple_qa import SimpleQA

TRAINING_SCRIPT_ROOT = Path(__file__).absolute().parent.parent
SIMPLE_QA_QUESTIONS_PATH = os.path.join(TRAINING_SCRIPT_ROOT, "resources", 'questionset.csv')

if __name__ == "__main__":
    model_repo=''

    file_path = './model_eval_scores.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(f'model_repo,reward_score,alignment_score,simple_qa_score\n')

    #load model and tokenizer
    tokenizer = get_tokenizer(model_repo)
    model = get_base_model(model_repo, tokenizer)

    #load evaluation model and tokenizer
    reward_model, reward_tokenizer = get_reward_model_and_tokenizer(REWARD_MODEL_REPO)
    alignment_model, alignment_tokenizer = get_reward_model_and_tokenizer(ALIGNMENT_MODEL_REPO)
    print("Reward models and tokenizers loaded successfully!")

    #load datasets
    eval_dataset = load_data(FULL_EVAL_DATASET)
    questions = pd.read_csv(SIMPLE_QA_QUESTIONS_PATH)

    rm = RewardMetrics(
        dataset=eval_dataset,
        model=model,
        reward_model=reward_model,
        alignment_model=alignment_model,
        tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        alignment_tokenizer=alignment_tokenizer,
        verbose=True
    )

    sqa = SimpleQA(
        dataset=questions,
        model=model,
        tokenizer=tokenizer,
        attempts=1, 
        verbose=True, 
        record_answers=True, 
        save_path=f'./simpleQA_outputs/{SUBMISSION_ID}.csv'
    )

    reward_score, alignment_score = rm.calculate_scores()
    simple_qa_score = sqa.calculate_simple_qa_score()

    with open(file_path, 'a') as file:
        file.write(f'{model_repo},{reward_score},{alignment_score},{simple_qa_score}\n')

    print(f"model_repo: {model_repo}; reward score: {reward_score}; alignment score: {alignment_score}; simple qa score: {simple_qa_score}\n")
