import json
import os
import csv
import requests
import pandas as pd
import numpy as np

from datasets import Dataset, load_dataset, concatenate_datasets

VALID_PAIR_EVAL_DATASETS = ["auto_j", "instrusum", "hhh", "preference_bench", "eval_bias_bench", "lfqa_eval"]
VALID_POINT_EVAL_DATASETS = ["flask", "mt_bench", "feedback_bench", "biggen_bench"]
VALID_CLASS_EVAL_DATASETS = ["llm-aggrefact", "info_bench_expert"]

MAX_RETRIES = 10 # max num times to try and get dataset from github sources via requests
RETRY_DELAY = 10 # seconds before next retry

INSTRUSUM_INPUT_TEMPLATE="""
Here is an article:
{article}

Here is an summary requirement:
{requirement}

Please summarize the above article based on the given requirement:
""".strip()

FEEDBACKBENCH_RUBRIC_TEMPLATE = """
[{criteria}]
    Score 1: {score_1}
    Score 2: {score_2}
    Score 3: {score_3}
    Score 4: {score_4}
    Score 5: {score_5}            
""".strip()

BIGGENBENCH_RUBRIC_TEMPLATE = """
[{orig_criteria}]
    Score 1: {orig_score1_description}
    Score 2: {orig_score2_description}
    Score 3: {orig_score3_description}
    Score 4: {orig_score4_description}
    Score 5: {orig_score5_description}
""".strip()


def get_dataset_from_github(url, is_jsonl=True):
    for attempt in range(MAX_RETRIES):
        response = requests.get(url)
        if response.status_code == 200:
            if is_jsonl:
                raw_data = response.text.splitlines()
                raw_dataset = [json.loads(line) for line in raw_data]
                return raw_dataset
            else:
                raw_data = json.loads(response.text)
                return raw_data
                
        else:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed with status code: {response.status_code}")
            time.sleep(RETRY_DELAY)
    
    return None

def all_agree(votes):
    yes_count = votes.count('yes')
    no_count = votes.count('no')
    
    if yes_count == len(votes):
        return 1
    elif no_count == len(votes):
        return 0
    else: 
        return None

def get_and_process_infobench(url):
    df = pd.read_csv(url)
    #df = df.replace(np.nan, "")
    data_in = df.to_dict(orient='records')

    data_out = []
    num_disagreements = 0
    models = ['gpt-3.5-turbo', 'gpt-4', 'claude-v1', 'alpaca-7b', 'vicuna-13b']

    for didx, d in enumerate(data_in):
        instruction_final = d['instruction']
        if isinstance(d['input'], str) and d['input'] != '':
            inst_input = d['input']
            instruction_final += f"\n{inst_input}"

        if not isinstance(d['decomposed_questions'], str):
            continue

        questions = [s[2:].strip() for s in d['decomposed_questions'].split('\n')]
        
        for model in models:
            if d[model] is not None:
                ann1_k, ann2_k = f"{model}-annotation-annotator1", f"{model}-annotation-annotator2"
                ann_k = f"{model}-annotation"

                
                ann1_ans, ann2_ans, ann_ans = [], [], []
                zip_lists = []
                if d[ann1_k] != '' and isinstance(d[ann1_k], str):
                    ann1_ans = ['yes' if 'yes' in s.lower() else 'no' for s in d[ann1_k].split('\n')]
                    zip_lists.append(ann1_ans)

                if d[ann2_k] != '' and isinstance(d[ann2_k], str):            
                    ann2_ans = ['yes' if 'yes' in s.lower() else 'no' for s in d[ann2_k].split('\n')]
                    zip_lists.append(ann2_ans)

                if d[ann_k] != '' and isinstance(d[ann_k], str):
                    ann_ans = ['yes' if s == '1' else 'no' for s in d[ann_k].split('.')]
                    zip_lists.append(ann_ans)
                
                # less than 3 experts answered
                if len(zip_lists) != 3:
                    continue
            
                for i, all_ans in enumerate(zip(*zip_lists)):
                    label_all_agree = all_agree(all_ans)

                    if label_all_agree != None:
                        output = {
                            'model': model,
                            'instruction': instruction_final,
                            'response': d[model],
                            'question': questions[i],
                            'label': label_all_agree,
                            'all_labels': all_ans
                        }
                        data_out.append(output)
                    else:
                        num_disagreements += 1

    return data_out


def load_eval_dataset(eval_dataset):
    # huggingface datasets
    if eval_dataset == 'auto_j':
        url = 'https://raw.githubusercontent.com/GAIR-NLP/auto-j/refs/heads/main/data/test/testdata_pairwise.jsonl'
        raw_dataset = get_dataset_from_github(url)
        if raw_dataset is not None:
            hf_dataset = Dataset.from_list(raw_dataset)
            return hf_dataset
        else:
            return None
    elif eval_dataset == 'lfqa_eval':
        url='https://github.com/carriex/lfqa_eval/raw/refs/heads/main/preference_data/experts_pairwise_human_preferences.jsonl'
        raw_dataset = get_dataset_from_github(url)
        if raw_dataset is not None:
            hf_dataset = Dataset.from_list(raw_dataset)

            def process_example(example):
                label = 1 if example['overall_preference'] == -1 else 2
                output = {
                    'label': label,
                }
                return output

            hf_dataset = hf_dataset.map(process_example, num_proc=10)
            hf_dataset = hf_dataset.select_columns(['question', 'answer_a', 'answer_b', 'label'])

            return hf_dataset
        else:
            return None

    elif eval_dataset == 'eval_bias_bench':
        url = 'https://raw.githubusercontent.com/ncsoft/offsetbias/refs/heads/master/data/evalbiasbench/biasbench.json'
        raw_dataset = get_dataset_from_github(url, is_jsonl=False)

        ds_formatted_out = {}

        if raw_dataset is not None:
            for split_name, data in raw_dataset.items():
                hf_dataset = Dataset.from_list(data)
                split_name_save = split_name.replace(' ', '_').strip()
                ds_formatted_out[split_name_save] = hf_dataset
             
            return ds_formatted_out
        else:
            return None


    elif eval_dataset == 'instrusum':
        # column_names = ["score_1", "sys_1", "doc_id", "requirement", "output_2", "winner", "sys_2", "output_1", "score_2", "article"]
        ds_raw = load_dataset("Salesforce/InstruSum", data_files='human_eval_pairwise.json', split='train')
        
        def process_example(example):
            

            input_text = INSTRUSUM_INPUT_TEMPLATE.format(
                article = example['article'].strip(),
                requirement = example['requirement'].strip(),
            )

            output = {
                'input': input_text,
            }

            return output

        ds_formatted = ds_raw.map(process_example, num_proc=10)
        ds_formatted = ds_formatted.rename_column("winner", "label")
        ds_formatted = ds_formatted.select_columns(['input', 'output_1', 'output_2', 'label'])

        return ds_formatted
    
    elif eval_dataset == 'hhh':
        splits = ["harmless", "helpful", "honest", "other"]

        ds_formatted_out = {}

        def process_example(example):
            responses = example['targets']
            output = {
                'input': example['input'],
                'output_1': responses['choices'][0],
                'output_2': responses['choices'][1],
                'label': 1 if responses['labels'][0] == 1 else 0
            }

            return output

        for split in splits:
            ds_raw = load_dataset('HuggingFaceH4/hhh_alignment', split)['test']
            ds_formatted = ds_raw.map(process_example, num_proc=10)
            ds_formatted = ds_formatted.select_columns(['input', 'output_1', 'output_2', 'label'])
            ds_formatted_out[split] = ds_formatted

        return ds_formatted_out
    elif eval_dataset == 'preference_bench':
        ds_raw = load_dataset('prometheus-eval/Preference-Bench', split='train')
        ds_formatted = ds_raw.select_columns(["orig_response_A", "orig_response_B", "orig_instruction", "orig_reference_answer", "orig_criteria", "orig_preference"])

        return ds_formatted

    elif eval_dataset == 'flask':
        url = 'https://raw.githubusercontent.com/prometheus-eval/prometheus-eval/refs/heads/main/eval/benchmark/data/flask_eval.json'
        raw_dataset = get_dataset_from_github(url, is_jsonl=False)
        
        if raw_dataset is not None:
            # FLASK raw data is parsed into prometheus prompt template. We extract the individual components
            extracted_raw_dataset = []
            for d in raw_dataset:
                input_orig = d['instruction'].split('###The instruction to evaluate:')[-1].split('###Response to evaluate:')[0].strip()
                response = d['instruction'].split('###Response to evaluate:')[-1].split('###Reference Answer (Score 5):')[0].strip()
                reference_answer = d['instruction'].split('###Reference Answer (Score 5):')[-1].split('###Score Rubrics:')[0].strip()
                rubric = d['instruction'].split('###Score Rubrics:')[-1].split('###Feedback:')[0].strip()

                output = {
                    'input': input_orig,
                    'response': response,
                    'reference_answer': reference_answer,
                    'rubric': rubric,
                    'human_score': d['human_score'],
                    'gpt4_score': d['gpt4_score']
                }
                
                extracted_raw_dataset.append(output)

            hf_dataset = Dataset.from_list(extracted_raw_dataset)
            return hf_dataset
        else:
            return None

    elif eval_dataset == 'mt_bench':
        url = 'https://raw.githubusercontent.com/prometheus-eval/prometheus-eval/refs/heads/main/eval/benchmark/data/mt_bench_eval.json'
        raw_dataset = get_dataset_from_github(url, is_jsonl=False)
        if raw_dataset is not None:
            # MT Bench raw data is parsed into prometheus prompt template. We extract the individual components
            extracted_raw_dataset = []
            for d in raw_dataset:
                input_orig = d['instruction'].split('###The instruction to evaluate:')[-1].split('###Response to evaluate:')[0].strip()
                response = d['instruction'].split('###Response to evaluate:')[-1].split('###Reference Answer (Score 5):')[0].strip()
                reference_answer = d['instruction'].split('###Reference Answer (Score 5):')[-1].split('###Score Rubrics:')[0].strip()
                rubric = d['instruction'].split('###Score Rubrics:')[-1].split('###Feedback:')[0].strip()

                output = {
                    'input': input_orig,
                    'response': response,
                    'reference_answer': reference_answer,
                    'rubric': rubric,
                    'gpt4_score': d['gpt4_score'],
                    'gpt4_feedback': d['gpt4_feedback']
                }
        
                
                extracted_raw_dataset.append(output)

            hf_dataset = Dataset.from_list(extracted_raw_dataset)
            return hf_dataset
        else:
            return None

    elif eval_dataset == 'feedback_bench':
        ds_raw = load_dataset('prometheus-eval/Feedback-Bench', split='train')
        
        def process_example(example):

            rubric = FEEDBACKBENCH_RUBRIC_TEMPLATE.format(
                criteria = example['orig_criteria'],
                score_1 = example['orig_score1_description'],
                score_2 = example['orig_score2_description'],
                score_3 = example['orig_score3_description'],
                score_4 = example['orig_score4_description'],
                score_5 = example['orig_score5_description'],
            ).strip()

            output = {
                'rubric': rubric,
                'gpt4_score': int(example['orig_score'])
            }

            return output

        ds_formatted = ds_raw.map(process_example, num_proc=10)
        ds_formatted = ds_formatted.select_columns(['orig_reference_answer', "orig_instruction", "orig_response", "rubric", "gpt4_score"])
        ds_formatted = ds_formatted.rename_column('orig_reference_answer', 'reference_answer')
        ds_formatted = ds_formatted.rename_column('orig_instruction', 'input')
        ds_formatted = ds_formatted.rename_column('orig_response', 'response')

        return ds_formatted
    
    elif eval_dataset == 'biggen_bench':
        data_files = {
            "human_eval": "data/human_eval-00000-of-00001.parquet",
            "llm_as_a_judge": "data/llm_as_a_judge-*.parquet",
            "multilingual_llm_as_a_judge": "data/multilingual_llm_as_a_judge-00000-of-00001.parquet",
            "multilingual_human_eval": "data/multilingual_human_eval-00000-of-00001.parquet"
            
        }
        ds_human_eval = load_dataset('prometheus-eval/BiGGen-Bench-Results', split='human_eval', data_files=data_files)
        ds_ml_human_eval = load_dataset('prometheus-eval/BiGGen-Bench-Results', split='multilingual_human_eval', data_files=data_files)

        ds_concat = concatenate_datasets([ds_human_eval, ds_ml_human_eval])

        def process_example(example):
            example_rubric = example['score_rubric']
            
            rubric = BIGGENBENCH_RUBRIC_TEMPLATE.format(
                orig_criteria = example_rubric['criteria'],
                orig_score1_description = example_rubric['score1_description'],
                orig_score2_description = example_rubric['score2_description'],
                orig_score3_description = example_rubric['score3_description'],
                orig_score4_description = example_rubric['score4_description'],
                orig_score5_description = example_rubric['score5_description'],
            )

            output = {
                'rubric': rubric
            }

            return output

        ds_formatted = ds_concat.map(process_example, num_proc=10)
        ds_formatted = ds_formatted.select_columns(['input', 'response', 'reference_answer', 'rubric', 'human_score', 'gpt4_score'])
        return ds_formatted

    elif eval_dataset == 'llm-aggrefact':
        ds_raw = load_dataset('lytang/LLM-AggreFact', revision='29e308a0c0c8af012943b70293dfb937811f13c6', split='test') # Pre-August 9, 2024 update
        ds_formatted = ds_raw.select_columns(['doc', 'claim', 'label'])
        return ds_formatted

    elif eval_dataset == 'info_bench_expert':
        url = 'https://drive.google.com/uc?id=1IKIRSLR3aPnBLhTd99nO09QQ72qiyKZc'
        data_out_expert_easy = get_and_process_infobench(url)

        url = 'https://drive.google.com/uc?id=161wLlIQzuHofbgkVvvSIn8cH5y6f4jlk'
        data_out_expert_hard = get_and_process_infobench(url)

        data_cat = data_out_expert_hard + data_out_expert_easy

        print(len(data_cat))

        hf_dataset = Dataset.from_list(data_cat)
        return hf_dataset