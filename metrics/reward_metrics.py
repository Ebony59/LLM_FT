import torch
import numpy as np
from tqdm import tqdm
import time

class RewardMetrics:
    def __init__(
        self,
        dataset,
        model,
        reward_model,
        alignment_model,
        tokenizer,
        reward_tokenizer,
        alignment_tokenizer,
        verbose=False
    ):
        self.dataset = dataset
        self.model = model
        self.reward_model = reward_model
        self.alignment_model = alignment_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.alignment_tokenizer = alignment_tokenizer
        self.verbose = verbose

    def calculate_scores(self):
        start_time = time.time()
        input_texts = self.generate_response()
        reward_score = self.evaluate_with_model(input_texts, self.reward_model, self.reward_tokenizer)
        alignment_score = self.evaluate_with_model(input_texts, self.alignment_model, self.alignment_tokenizer)
        end_time = time.time()

        print(f"Reward Score: {reward_score}; Alignment Score: {alignment_score}")
        print(f"Time elapsed: {end_time - start_time}")
        
        return reward_score, alignment_score
        
    def generate_response(self):
        self.model.eval()
        input_texts = []

        print('Generating responses to be evaluted')
        for i in tqdm(range(len(self.dataset))):
            prompt = self.dataset[i]['payload']

            inputs = self.tokenizer(prompt + '####\n', return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.half().generate(
                    **inputs,
                    max_new_tokens=200,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.split('\n####\n')[1]
            generated_text = generated_text.split('\n')[0]
            
            prompt = prompt.strip('\n')
            reward_input_text = f"{prompt} {generated_text}"

            if self.verbose:
                #print the response for a dataset 
                if i == 2:
                    print(reward_input_text)
            
            input_texts.append(reward_input_text)            

        return input_texts

    def evaluate_with_model(self, input_texts, model, tokenizer):
        scores = []

        with torch.no_grad():
            for i in tqdm(range(len(input_texts))):
                tokenised_inputs = tokenizer(input_texts[i], return_tensors="pt", padding=True, truncation=True).to("cuda")
                outputs = model(**tokenised_inputs)
                if outputs.logits.shape == torch.Size([1, 1]):
                    score = outputs.logits.item()
                else:
                    score = outputs.logits[:, 1].item()

                scores.append(score)

        score = np.mean(scores)
            
        return score