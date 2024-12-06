import retry
import re
import time
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, List
from openai import OpenAI

from metrics.templates import GRADER_TEMPLATE, QA_TEMPLATE

class SimpleQA:
    def __init__(
        self, 
        dataset, 
        model, 
        tokenizer, 
        attempts=1, 
        verbose=False, 
        record_answers=False, 
        save_path='./simpleQA_outputs/simpleQA_output.csv',
        generation_params={'temperature':0.8},
    ):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.attempts = attempts
        self.verbose = verbose
        self.record_answers = record_answers
        self.save_path = save_path
        self.generation_params = generation_params

    def calculate_simple_qa_score(self):
        start_time = time.time()
        self.generate_answer()
        self.get_grades()
        self.dataset['score'] = self.dataset['judge_response'].apply(lambda x: self.get_scores(x))
        end_time = time.time()
        score = round(self.dataset.score.mean(), 2)

        print(f"Simple QA Score: {score}")
        print(f"Time elapsed: {end_time - start_time}")
        if self.record_answers:
            self.dataset.to_csv(self.save_path)

        return score

    def generate_answer(self):
        #takes approx 7 min
        self.model.eval()
        output_texts = []
        predicted_answers = []

        print('Generating responses to be evaluted')
        for i in tqdm(range(len(self.dataset))):
            problem = self.dataset.loc[i,'problem']

            prompt = QA_TEMPLATE + f'user: {problem}\n'

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            generated_texts = []
            raw_output = []
            
            for attempt in range(self.attempts):
                with torch.no_grad():
                    outputs = self.model.half().generate(
                        **inputs,
                        max_new_tokens=200,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.8,
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                raw_output.append(generated_text)

                generated_text = generated_text[len(prompt):]
                generated_texts.append(generated_text)
            
            output_texts.append(raw_output)
            predicted_answers.append(generated_texts)

            if self.verbose:
                #print the response for the first question
                if i == 0:
                    print("Question:", problem)
                    print("Answers:", generated_texts)
        
        self.dataset['output_text'] = output_texts
        self.dataset['predicted_answer'] = predicted_answers

    def get_grades(self):
        #approx 2 min
        grades = []

        print("Grading the responses")
        for i in tqdm(range(len(self.dataset))):
            problem = self.dataset.loc[i, 'problem']
            answer = self.dataset.loc[i, 'answer']
            predicted_answer = self.dataset.loc[i, 'predicted_answer']
            
            prompt = GRADER_TEMPLATE.format(question=problem, target=answer)
            formatted_answer = f"user: There are {len(predicted_answer)} predicted answers for the questsion in the following list:\n{predicted_answer}\n"
            
            messages = [
                {
                    'role': 'system',
                    'content': GRADER_TEMPLATE.format(question=problem,
                                                    target=answer)
                },
                {
                    'role': 'user',
                    'content': f"There are {len(predicted_answer)} predicted answers for the questsion in the following list:\n{predicted_answer}"
                }
            ]
            generation_params = {'temperature': 0.8}

            class Score(BaseModel):
                score: List[Literal['A', 'B', 'C']] = Field(..., description="The score of the evaluation. A is correct answer, B is incorrect answer, C is not attempted answer")

            llm = OpenAI()
            grading_response = llm.beta.chat.completions.parse(
                model = "gpt-4o-2024-08-06",
                messages = messages,
                response_format = Score,
            ).choices[0].message.content
            grade = json.loads(grading_response)['score']
            grades.append(grade)
            
        self.dataset['judge_response'] = grades

    def get_scores(self, grades):
        score_map = {'A': 1, 'B': -1, 'C': -0.6, 'D': -2}
        scores = []

        if len(grades) != self.attempts:
            return 0
        
        for item in grades:
            scores.append(score_map.get(item, 0))

        return np.mean(scores)
