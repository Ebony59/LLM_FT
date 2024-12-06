import torch
import wandb
import numpy as np
from tqdm import tqdm
import time

from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    pipeline,
)

from metrics.reward_metrics import RewardMetrics

class RewardLoggingCallback(TrainerCallback):
    def __init__(
        self, 
        dataset, 
        reward_model, 
        alignment_model,
        tokenizer,
        reward_tokenizer, 
        alignment_tokenizer, 
        eval_steps=100, 
        verbose=False,
    ):
        self.dataset = dataset
        self.reward_model = reward_model
        self.alignment_model = alignment_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.alignment_tokenizer = alignment_tokenizer
        self.eval_steps = eval_steps
        self.verbose = verbose

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps == 0:
            
            rm = RewardMetrics(
                dataset=self.dataset,
                model=kwargs['model'],
                reward_model=self.reward_model,
                alignment_model=self.alignment_model,
                tokenizer=self.tokenizer,
                reward_tokenizer=self.reward_tokenizer,
                alignment_tokenizer=self.alignment_tokenizer,
                verbose=self.verbose
            )

            reward_score, alignment_score = rm.calculate_scores()
            print(f"Step {state.global_step}: Reward Score: {reward_score}; Alignment Score: {alignment_score}")
            wandb.log({f"reward scores": reward_score, "step": state.global_step})
            wandb.log({f"alignment scores": alignment_score, "step": state.global_step})