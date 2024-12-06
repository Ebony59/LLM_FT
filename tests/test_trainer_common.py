import os
import pytest
import vcr

from transformers import GPT2ForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from peft import PeftModel
from mock import patch, MagicMock
from datasets import *
from chaiverse import ModelSubmitter
from chaiverse.formatters import PygmalionFormatter

from trainers import common


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')
filtered_vcr = vcr.VCR(filter_headers = ["Authorization", "developer_key"])


@pytest.fixture
def ds():
    with filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'ds.yaml')):
        return common.load_data('ChaiML/horror_data_formatted')


@pytest.fixture
def generation_params():
    return {
            'frequency_penalty': 0.5,
            'max_input_tokens': 1024,
            'presence_penalty': 0.5,
            'stopping_words': ['\n'],
            'temperature': 0.9,
            'top_k': 80,
            'top_p': 0.95,
            'min_p': 0.05,
            'best_of': 4,
        }


def test_load_data(ds):
    assert isinstance(ds, Dataset)
    assert len(ds)>0


def test_get_tokenizer(tiny_tokenizer):
    assert isinstance(tiny_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))


def test_get_base_model(tiny_base_model):
    assert isinstance(tiny_base_model, PreTrainedModel)


def test_print_trainable_parameters(tiny_base_model, capsys):
    common.print_trainable_parameters(tiny_base_model)
    captured = capsys.readouterr()

    expected_output = "Total number of trainable parameters"

    assert expected_output in captured.out


def test_get_lora_model(tiny_lora_model):
    assert isinstance(tiny_lora_model, PeftModel)


def test_get_reward_model_and_tokenizer(tiny_reward_model_and_tokenizer):
    tiny_reward_model, tiny_reward_tokenizer = tiny_reward_model_and_tokenizer
    assert isinstance(tiny_reward_model, GPT2ForSequenceClassification)
    assert isinstance(tiny_reward_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))


def test_merge_and_upload_model(tiny_lora_model, tiny_tokenizer):
    model_repo = 'dummy-repo'
    
    with patch.object(tiny_tokenizer, "push_to_hub") as mock_tokenizer_push, \
         patch.object(tiny_lora_model.merge_and_unload(), "push_to_hub") as mock_model_push:
        
        trained_model = common.merge_and_upload_model(tiny_lora_model, tiny_tokenizer, model_repo)

        mock_tokenizer_push.assert_called_once_with(model_repo, private=True)
        mock_model_push.assert_called_once_with(model_repo, private=True)

        assert trained_model == tiny_lora_model.merge_and_unload()


def test_submit_to_chaiverse_with_memories(generation_params):
    model_repo = 'dummy-repo'

    with patch.object(ModelSubmitter, "submit") as mock_submit:
        common.submit_to_chaiverse(model_repo, generation_params)


        submission_parameters = {
            "model_repo": model_repo,
            "generation_params": generation_params,
        }

        mock_submit.assert_called_once_with(submission_parameters)


def test_submit_to_chaiverse_without_memories(generation_params):
    model_repo = 'dummy-repo'

    with patch.object(ModelSubmitter, "submit") as mock_submit:
        common.submit_to_chaiverse(model_repo, generation_params, include_memories=False)

        formatter = PygmalionFormatter()
        formatter.memory_template = ''
        formatter.prompt_template = ''

        submission_parameters = {
            "model_repo": model_repo,
            "generation_params": generation_params,
            "formatter": formatter
        }

        mock_submit.assert_called_once_with(submission_parameters)
    
    
    