import pytest
import os
import vcr
import pandas as pd
from mock import patch, MagicMock
from metrics.simple_qa import SimpleQA
from metrics.templates import QA_TEMPLATE

RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')
filtered_vcr = vcr.VCR(filter_headers = ["Authorization", "api_key"])


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


def test_simpleQA_generate_answer(questions, tiny_base_model, tiny_tokenizer):
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
    )

    dummy_generated_tokens = [[250, 250, 250, 250]]
    problem = questions.loc[0,'problem']
    prompt = QA_TEMPLATE + f'user: {problem}\n'
    dummy_decoded_text = prompt + '1997'
    
    with patch.object(sqa.model, 'generate', return_value=dummy_generated_tokens) as mock_generate, \
         patch.object(sqa.tokenizer, "decode", return_value=dummy_decoded_text) as mock_text:
        
        sqa.generate_answer()

        assert mock_generate.call_count == len(questions)
        assert mock_text.call_count == len(questions)

        assert sqa.dataset.loc[0, 'predicted_answer'] == ['1997']


@filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_simpleQA_get_grades.yaml'))
def test_simpleQA_get_grades(questions, tiny_base_model, tiny_tokenizer):  
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
    )

    sqa.dataset['predicted_answer'] = [['1997.'], ['The Two Ultraman.'],['I can not answer.']]
    sqa.get_grades()

    assert sqa.dataset.loc[0, 'judge_response'] == ['A']
    assert sqa.dataset.loc[1, 'judge_response'] == ['B']
    assert sqa.dataset.loc[2, 'judge_response'] == ['C']


@filtered_vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_simpleQA_get_multiple_grades.yaml'))
def test_simpleQA_get_multiple_grades(questions, tiny_base_model, tiny_tokenizer):  
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
        attempts=2,
    )

    sqa.dataset['predicted_answer'] = [['1997.','Is 1997'], ['The Two Ultraman.', 'The Two Fridas.'],['I can not answer.', '1950']]
    sqa.get_grades()

    assert sqa.dataset.loc[0, 'judge_response'] == ['A', 'A']
    assert sqa.dataset.loc[1, 'judge_response'] == ['B', 'A']
    assert sqa.dataset.loc[2, 'judge_response'] == ['C', 'B']


def test_simpleQA_get_scores(questions, tiny_base_model, tiny_tokenizer):
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
    )
    sqa.dataset['predicted_answer'] = [['1997.'], ['The Two Ultraman.'],['I can not answer.']]
    sqa.dataset['judge_response'] = [['A'], ['B'],['C']]
    sqa.dataset['score'] = sqa.dataset['judge_response'].apply(lambda x: sqa.get_scores(x))

    assert sqa.dataset.loc[0, 'score'] == 1
    assert sqa.dataset.loc[1, 'score'] == -1
    assert sqa.dataset.loc[2, 'score'] == -0.6


def test_simpleQA_get_multiple_scores(questions, tiny_base_model, tiny_tokenizer):
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
        attempts=2,
    )
    sqa.dataset['predicted_answer'] = [['1997.','Is 1997'], ['The Two Ultraman.', 'The Two Fridas.'],['I can not answer.', '1950']]
    sqa.dataset['judge_response'] = [['A', 'A'], ['B', 'A'],['C', 'B']]
    sqa.dataset['score'] = sqa.dataset['judge_response'].apply(lambda x: sqa.get_scores(x))

    assert sqa.dataset.loc[0, 'score'] == 1
    assert sqa.dataset.loc[1, 'score'] == 0
    assert sqa.dataset.loc[2, 'score'] == -0.8
    

def test_simpleQA_calculate_simple_qa_score(questions, tiny_base_model, tiny_tokenizer, capsys):
    sqa = SimpleQA(
        dataset=questions, 
        model=tiny_base_model,
        tokenizer=tiny_tokenizer,
    )
    sqa.dataset['predicted_answer'] = [['1997.'], ['The Two Ultraman.'],['I can not answer.']]
    sqa.dataset['judge_response'] = [['A'], ['B'],['C']]

    with patch.object(sqa, "generate_answer") as mock_generate, \
         patch.object(sqa, "get_grades") as mock_get_grades:

        score = sqa.calculate_simple_qa_score()
        
        assert score == round((1-1-0.6)/3, 2)
    
        captured = capsys.readouterr()

        assert f"Simple QA Score: {score}" in captured.out
        assert "Time elapsed" in captured.out
        
        