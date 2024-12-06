import pytest
import vcr
import pandas as pd
import os
from mock import patch, MagicMock
from datasets import *

from transformers import TrainerState, TrainerControl
from callbacks.reward_callback import RewardLoggingCallback
from callbacks.simple_qa_callback import SimpleQALoggingCallback


@pytest.fixture
def dataset():
    data = {
        "payload": [
            "Tom: Tom is your friend.\nUser: Hi.\nTom:\n",
            "Albert: Albert is your friend.\nUser: Hi!\nAlbert:\n"
        ]
    }
    df = pd.DataFrame(data=data)
    return Dataset.from_pandas(df)


@pytest.fixture
def questions():
    data = {
        'problem' : [
            'During what year did Pipilotti Rist receive the Premio 2000 prize?', 
            "What is Frida Kahlo's largest painting called in English?",
            "Which year was Bil Keane's first syndicated strip, Channel Chuckles, launched?"
        ],
        'answer' : ['1997', 'The Two Fridas.','1954'],
    }
    return pd.DataFrame(data=data)


def test_reward_logging_callback(dataset, tiny_lora_model, tiny_reward_model_and_tokenizer, tiny_tokenizer):
    tiny_reward_model, tiny_reward_tokenizer = tiny_reward_model_and_tokenizer
    
    reward_logging_callback = RewardLoggingCallback(
        dataset=dataset,
        reward_model=tiny_reward_model,
        alignment_model=tiny_reward_model,
        tokenizer=tiny_tokenizer,
        reward_tokenizer=tiny_reward_tokenizer,
        alignment_tokenizer=tiny_reward_tokenizer,
        eval_steps=100,
        verbose=True,
    )
    
    state = TrainerState()
    control = TrainerControl()

    state.global_step = 200

    mock_reward_metrics = MagicMock()
    mock_reward_metrics_instance = MagicMock()
    mock_reward_metrics.return_value = mock_reward_metrics_instance
    mock_reward_metrics_instance.calculate_scores.return_value = (0.8, 0.9)
    
    with patch("callbacks.reward_callback.RewardMetrics", mock_reward_metrics), \
         patch("callbacks.reward_callback.wandb.log") as mock_wandb_log:
        reward_logging_callback.on_step_end(args=None, state=state, control=control, model=tiny_lora_model)
    
        mock_reward_metrics.assert_called_once_with(
            dataset=dataset,
            model=tiny_lora_model,
            reward_model=tiny_reward_model,
            alignment_model=tiny_reward_model,
            tokenizer=tiny_tokenizer,
            reward_tokenizer=tiny_reward_tokenizer,
            alignment_tokenizer=tiny_reward_tokenizer,
            verbose=True
        )

        mock_reward_metrics_instance.calculate_scores.assert_called_once()

        mock_wandb_log.assert_any_call({"reward scores": 0.8, "step": 200})
        mock_wandb_log.assert_any_call({"alignment scores": 0.9, "step": 200})


def test_reward_logging_callback_not_called(dataset, tiny_lora_model, tiny_reward_model_and_tokenizer, tiny_tokenizer):
    tiny_reward_model, tiny_reward_tokenizer = tiny_reward_model_and_tokenizer
    
    reward_logging_callback = RewardLoggingCallback(
        dataset=dataset,
        reward_model=tiny_reward_model,
        alignment_model=tiny_reward_model,
        tokenizer=tiny_tokenizer,
        reward_tokenizer=tiny_reward_tokenizer,
        alignment_tokenizer=tiny_reward_tokenizer,
        eval_steps=100,
        verbose=True,
    )
    
    state = TrainerState()
    control = TrainerControl()

    state.global_step = 250
    
    with patch("callbacks.reward_callback.wandb.log") as mock_wandb_log:
        reward_logging_callback.on_step_end(args=None, state=state, control=control, model=tiny_lora_model)

        mock_wandb_log.assert_not_called()


def test_simpleqa_logging_callback(questions, tiny_lora_model, tiny_tokenizer):
    simple_qa_callback = SimpleQALoggingCallback(
        tokenizer=tiny_tokenizer,
        dataset=questions
    )

    state = TrainerState()
    control = TrainerControl()

    state.global_step = 200

    mock_simple_qa = MagicMock()
    mock_simple_qa_instance = MagicMock()
    mock_simple_qa.return_value = mock_simple_qa_instance
    mock_simple_qa_instance.calculate_simple_qa_score.return_value = -0.5

    with patch("callbacks.simple_qa_callback.SimpleQA", mock_simple_qa), \
         patch("callbacks.simple_qa_callback.wandb.log") as mock_wandb_log:
        simple_qa_callback.on_step_end(args=None, state=state, control=control, model=tiny_lora_model)

        mock_simple_qa.assert_called_once_with(
            dataset=questions,
            model=tiny_lora_model,
            tokenizer=tiny_tokenizer,
            attempts=1,
            verbose=False,
            record_answers=False,
            save_path='./simpleQA_outputs/simpleQA_score_step_200.csv'
        )

        mock_simple_qa_instance.calculate_simple_qa_score.assert_called_once()

        mock_wandb_log.assert_any_call({"simpleQA scores": -0.5, "step": 200})
    
    
def test_simpleqa_logging_callback_not_called(questions, tiny_lora_model, tiny_tokenizer):
    simple_qa_callback = SimpleQALoggingCallback(
        tokenizer=tiny_tokenizer,
        dataset=questions
    )

    state = TrainerState()
    control = TrainerControl()

    state.global_step = 250

    with patch("callbacks.simple_qa_callback.wandb.log") as mock_wandb_log:
        simple_qa_callback.on_step_end(args=None, state=state, control=control, model=tiny_lora_model)

        mock_wandb_log.assert_not_called()