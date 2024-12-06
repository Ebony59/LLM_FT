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

from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    pipeline,
)

from metrics.simple_qa import SimpleQA

class SimpleQALoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, attempts=1, eval_steps=100, verbose=False, record_answers=False, save_dir='./simpleQA_outputs'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.eval_steps = eval_steps
        self.attempts = attempts
        self.verbose = False
        self.record_answers = record_answers
        self.save_dir = save_dir

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps == 0:
            save_path = f'{self.save_dir}/simpleQA_score_step_{state.global_step}.csv'
            
            sqa = SimpleQA(
                dataset=self.dataset, 
                model=kwargs['model'],
                tokenizer=self.tokenizer,
                attempts=self.attempts,
                verbose=self.verbose, 
                record_answers=self.record_answers,
                save_path=save_path
            )
            score = sqa.calculate_simple_qa_score()

            wandb.log({f"simpleQA scores": score, "step": state.global_step})
            print(f"Step {state.global_step}: Simple QA Score: {score};")
       