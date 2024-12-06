from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model
import torch

def load_data(dataset_name):
    ds = load_dataset(dataset_name, split='train')
    return ds

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = 'left'
    return tokenizer

def get_base_model(model_id, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model

def print_trainable_parameters(model):
    size = 0
    for name, param in model.named_parameters():
      if param.requires_grad:
          size += param.size().numel()
    print(f'Total number of trainable parameters: {size // 1e6} million')


def get_lora_model(model, lora_config):
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model


def get_reward_model_and_tokenizer(model_repo):
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_repo).to("cuda")
    reward_tokenizer = AutoTokenizer.from_pretrained(model_repo)
    if not reward_tokenizer.pad_token:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    return reward_model, reward_tokenizer


def merge_and_upload_model(lora_model, tokenizer, model_repo):
    trained_model = lora_model.merge_and_unload()
    tokenizer.push_to_hub(model_repo, private=True)
    trained_model.push_to_hub(model_repo, private=True)
    return trained_model
