import os
import vcr
import pytest
from trainers import common

from peft import LoraConfig


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')
filtered_vcr = vcr.VCR(
    filter_headers = ["Authorization", "developer_key", "X-Amzn-Trace-Id"],
    match_on = ["scheme", "host", "port", "path", "query"],
)


@pytest.fixture(scope='session')
def tiny_base_model_id():
    return "HuggingFaceM4/tiny-random-MistralForCausalLM"


@pytest.fixture(scope='session')
def tiny_reward_model_id():
    return "hf-tiny-model-private/tiny-random-GPT2ForSequenceClassification"


@pytest.fixture(scope='session')
def tiny_tokenizer(tiny_base_model_id):
    with filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'tiny_tokenizer.yaml')):
        return common.get_tokenizer(tiny_base_model_id)


@pytest.fixture(scope='session')
def tiny_base_model(tiny_base_model_id, tiny_tokenizer):
    with filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'tiny_base_model.yaml')):
        return common.get_base_model(tiny_base_model_id, tiny_tokenizer)


@pytest.fixture(scope='session')
def lora_config():
    return LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


@pytest.fixture(scope='session')
def tiny_lora_model(tiny_base_model, lora_config):
    return common.get_lora_model(tiny_base_model, lora_config)


@pytest.fixture(scope='session')
def tiny_reward_model_and_tokenizer(tiny_reward_model_id):
    with filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'tiny_reward_model.yaml')):
        return common.get_reward_model_and_tokenizer(tiny_reward_model_id)


@pytest.fixture(scope="session", autouse=True)
def mock_openai_api():
    if os.environ.get('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = "dummy-api-key"