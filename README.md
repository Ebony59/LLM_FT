# Training Scripts for DPO & SFT

## Getting ready...

### Install packages
`pip install -r requirements.txt`

run `pip install flash-attn==2.6.3 --no-build-isolation` separately

### login with huggingface, chaiverse, openAI
login with huggingface
`huggingface-cli login`

set your openai api key as environment variable
`export OPENAI_API_KEY="your-api-key-here`

### Add training_scipts to your pythonpath
`export PYTHONPATH=/path/to/training_script:$PYTHONPATH`

## Model and Datasets
Add the model and datasets in `config/repo_path.py`. 

## Run training scripts

SFT: `trainers/sft_trainer.py`

DPO: `trainers/dpo_trainer.py`

Reward and alignment scores evaluation is performed on 50 dataset, and takes approximately 8 minutes. SimpleQA scores are evaluated on 250 questions using gpt-4o as a judge, and takes around 11 minutes.

By default, reward and alignment scores are evaluated every half an epoch. SimpleQA scores are evaluated every epoch. It can be adjusted by changing `eval_steps` when calling `reward_logging_callback` and `simple_qa_logging_callback`. 

Note: steps per epoch is calculated as: len(training_dataset)/(per_device_train_batch_size * gradient_accumulation_steps). 

## Calculate the scores separately

You can evaluate the reward, alignment and simpleQA scores using: `metrics/calculate_scores.py`. This uses a larger evaluation dataset (100).
