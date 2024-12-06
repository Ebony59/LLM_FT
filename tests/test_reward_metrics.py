import pytest
import os
from mock import patch, MagicMock
import pandas as pd
import numpy as np
from datasets import *
from metrics.reward_metrics import RewardMetrics


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
def rm(dataset, tiny_base_model, tiny_tokenizer, tiny_reward_model_and_tokenizer):
    tiny_reward_model, tiny_reward_tokenizer = tiny_reward_model_and_tokenizer
    return RewardMetrics(
        dataset=dataset,
        model=tiny_base_model,
        reward_model=tiny_reward_model,
        alignment_model=tiny_reward_model,
        tokenizer=tiny_tokenizer,
        reward_tokenizer=tiny_reward_tokenizer,
        alignment_tokenizer=tiny_reward_tokenizer
    )


def test_reward_metrics_generate_response(rm):
    dummy_generated_token = [[250, 250, 250, 250]]
    prompts = [rm.dataset[i]['payload'] + "####\n" for i in range(len(rm.dataset))]
    generated_text = "Hello friend."
    dummy_decoded_text = prompts[0] + generated_text

    with patch.object(rm.model, "generate", return_value=dummy_generated_token) as mock_generate, \
         patch.object(rm.tokenizer, "decode", return_value=dummy_decoded_text) as mock_text:
        input_texts = rm.generate_response()
        
        assert input_texts == [rm.dataset[i]['payload'].strip('\n')+' '+generated_text for i in range(len(rm.dataset))]


def test_reward_metrics_evaluate_with_model(rm):
    input_texts = ["Tom: Tom is your friend.\nUser: Hi.\nTom: Hello friend.", "Albert: Albert is your friend.\nUser: Hi!\nAlbert: Hello friend."]
    reward_score = rm.evaluate_with_model(input_texts, rm.reward_model, rm.reward_tokenizer)
    
    assert np.isclose(reward_score, -0.009298, atol=0.01, rtol=0.01)
