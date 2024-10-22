import json
import os
import time
import argparse
import importlib
import csv
import glob
import string
import gc
import torch

from collections import defaultdict

from tqdm import tqdm
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment     # for shutting down vllm

from utils import compute_acc, compute_pointwise_metrics, compute_classification_metrics                # compute metrics
from utils import VALID_PAIR_EVAL_DATASETS, VALID_POINT_EVAL_DATASETS, VALID_CLASS_EVAL_DATASETS        # list of valid datasets
from prompt_utils import get_prompt, get_prompt_point, get_prompt_classification                     
from data_utils import load_eval_dataset # loads datasets

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

DATASET_TO_TASK = {
    "auto_j":               "pair",
    "instrusum":            "pair",
    "hhh":                  "pair",
    "preference_bench":     "pair",
    "eval_bias_bench":      "pair",
    "lfqa_eval":            "pair",
    "flask":                "point",
    "mt_bench":             "point",
    "feedback_bench":       "point",
    "biggen_bench":         "point",
    "llm-aggrefact":        "class",
    "info_bench_expert":    "class",
}

LLAMA2_TEMPLATE="[INST] {input} [/INST]" # for Auto-J


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn' #needed for vllm>=0.5.3.post1

def main(args):
    # Load model
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
    if 'mistralai' in args.model or 'prometheus' in args.model.lower():
        os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS' # needed for sliding window (mistral/prometheus)

    # For trained baseline models
    model_modifier = ''
    if args.sfr_prompt_template == 'task_specific':
        model_modifier = 'sfr_task_specific'
    elif args.sfr_prompt_template == 'prepair':
        model_modifier = 'sfr_prepair'
    elif args.sfr_prompt_template == 'rewardbench':
        model_modifier = 'sfr_rewardbench'
    elif 'prometheus' in args.model.lower():
        model_modifier = 'prometheus'
    elif 'offsetbias' in args.model.lower():
        model_modifier = 'offsetbias'
    elif 'autoj-13b' in args.model.lower():
        model_modifier = 'auto_j'
    elif 'skywork' in args.model.lower():
        model_modifier = 'skywork'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "llama-3" in args.model.lower() or "llama3" in args.model.lower():
        stop_token_ids = [128009]
    else:
        stop_token_ids = []

    print('-'*50)
    print(f'EVALUATING: {args.model}')
    print('-'*50)
    if 'nemo' in args.model.lower():
        model = LLM(args.model, trust_remote_code=True, tensor_parallel_size=args.num_gpus, max_model_len=131072)
    else:
        model = LLM(args.model, trust_remote_code=True, tensor_parallel_size=args.num_gpus)

    # Sampling parameters for outputs. Paper uses temperature = 0, top_p = 1, n = 1
    sampling_params = SamplingParams(
        n=args.num_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1024,
        stop_token_ids=stop_token_ids,
    )

    # Get list of all valid datasets
    VALID_EVAL_DATASETS = VALID_PAIR_EVAL_DATASETS + VALID_POINT_EVAL_DATASETS + VALID_CLASS_EVAL_DATASETS
    eval_dataset_in = args.eval_dataset

    # Based on input, select dataset to evaluate. Options are
    #   all:                        Evaluate all pair, pointwise (rating), and classification datasets
    #   all_{pair,point,class}:     Evaluation task specific
    #   {dataset_name}:             Evaluate only that dataset
    if eval_dataset_in == "all":
        eval_dataset_list = VALID_EVAL_DATASETS
    elif eval_dataset_in == 'all_pair':
        eval_dataset_list = VALID_PAIR_EVAL_DATASETS
    elif eval_dataset_in == 'all_point':
        eval_dataset_list = VALID_POINT_EVAL_DATASETS
    elif eval_dataset_in == 'all_class':
        eval_dataset_list = VALID_CLASS_EVAL_DATASETS
    else:
        assert eval_dataset_in in VALID_EVAL_DATASETS
        eval_dataset_list = [eval_dataset_in]

    # Iterate over all datasets
    for eval_dataset in eval_dataset_list:
        if eval_dataset == "rewardbench":
            print("Please use RewardBench repo to evaluate on RewardBench!")
            continue

        task_type = DATASET_TO_TASK[eval_dataset] # Get if pairwise, pointwise (rating), or classification task

        # Get appropriate prompt based on eval dataset and particular models
        if task_type == 'pair':
            prompt, prompt_name, get_prediction = get_prompt(eval_dataset, model_modifier)
        elif task_type == 'point':
            prompt, prompt_name, get_prediction = get_prompt_point(eval_dataset, model_modifier)
            if get_prediction is None:
                from prompt_utils import get_prediction_pointwise as get_prediction
        elif task_type == 'class':
            prompt, prompt_name, get_prediction = get_prompt_classification(eval_dataset, model_modifier)

        # Load dataset(s)
        output_path_template = args.output_path
        output_paths = []
        datasets = []

        # One file in eval set / benchmark has no subsets
        print(f"Loading dataset: {eval_dataset}")
        loaded_datasets = load_eval_dataset(eval_dataset)

        if loaded_datasets == None:
            print(f"Failed to load {eval_dataset}!")

         # Multiple subsets in benchmark, load each subset separately. load_dataset returns {split_name: hf_dataset} dict
        elif isinstance(loaded_datasets, dict):
            for split_name, generation_dataset in loaded_datasets.items():
                datasets.append(generation_dataset)
                output_path_format = output_path_template.format(
                    dataset_name=eval_dataset,
                    prompt=prompt_name,
                    signature=split_name
                )

                base_dir, fname = output_path_format.split(f'/eval_result/{eval_dataset}/') 
                output_path_format = base_dir + f'/eval_result/{eval_dataset}/{split_name}-' + fname

                output_paths.append(output_path_format)
        
        else:
            datasets.append(loaded_datasets)
            output_path_format = output_path_template.format(
                dataset_name=eval_dataset,
                prompt=prompt_name,
                signature=eval_dataset
            )
            output_paths.append(output_path_format)


        print(f'PROCESSING {eval_dataset} | TASK_TYPE: {task_type} | OUTPUT PATHS: {output_paths}')
        for generation_dataset, output_path in zip(datasets, output_paths):
            if args.debug:
                generation_dataset = generation_dataset.select(range(10))

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if task_type == 'pair':

                # For each dataset point, run eval twice, swapping order (consistency)
                def process_example(generation):
                    pair_inferences = []

                    # different dataset keys corresponding to responses for different datasets
                    if eval_dataset == "auto_j":
                        output_keys = [("response 1", "response 2"), ("response 2", "response 1")]
                    elif eval_dataset == "preference_bench":
                        output_keys = [("orig_response_A", "orig_response_B"), ("orig_response_B", "orig_response_A")]
                    elif eval_dataset == "eval_bias_bench":
                        output_keys = [("response1", "response2"), ("response2", "response1")]
                    elif eval_dataset == "lfqa_eval":
                        output_keys = [("answer_a", "answer_b"), ("answer_b", "answer_a")]
                    else:
                        output_keys = [("output_1", "output_2"), ("output_2", "output_1")]


                    for idx, pair in enumerate(output_keys): # iterate over order of responses (consistency)
                        prompt_input_args = [tup[1] for tup in string.Formatter().parse(prompt) if tup[1] is not None]

                        # Only preference bench has rubric and reference answers
                        if eval_dataset == 'preference_bench' and 'reference_answer' in prompt_input_args and 'rubric' in prompt_input_args:
                            content = prompt.format(
                                input=generation['orig_instruction'], 
                                output_1=generation[pair[0]], 
                                output_2=generation[pair[1]],
                                reference_answer=generation['orig_reference_answer'],
                                rubric=generation['orig_criteria']
                            ).strip()
                        
                        else:
                            # Get dataset-specific keys for original user input
                            if eval_dataset == 'auto_j':
                                input_key = 'prompt'
                            elif eval_dataset == 'preference_bench':
                                input_key = 'orig_criteria'
                            elif eval_dataset == 'eval_bias_bench':
                                input_key = "instruction"
                            elif eval_dataset == "lfqa_eval":
                                input_key = "question"
                            else:
                                input_key = 'input'

                            content = prompt.format(
                                input=generation[input_key], 
                                output_1=generation[pair[0]], 
                                output_2=generation[pair[1]]
                            ).strip()

                        messages = [
                            {"role": "user", "content": content}
                        ]

                        # Custom chat template for Auto-J following their implementation
                        # https://github.com/GAIR-NLP/auto-j/blob/main/codes/usage/constants_prompt.py
                        if 'gair' in args.model.lower(): 
                            generation[f'text_{idx+1}'] = LLAMA2_TEMPLATE.format(input=content)
                        # Apply chat templates
                        else:
                            input_seq = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            generation[f'text_{idx+1}'] = input_seq
                    
                    return generation

            elif task_type == 'point':
                def process_example(generation):
                    content = prompt.format(
                        instruction=generation['input'],
                        response=generation['response'],
                        reference_answer=generation['reference_answer'],
                        rubric = generation['rubric'] 
                    )
                            
                    messages = [
                        {"role": "user", "content": content}
                    ]

                    if 'gair' in args.model.lower():
                        generation[f'text'] = LLAMA2_TEMPLATE.format(input=content)
                    else:
                        input_seq = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        generation[f'text'] = input_seq
                    return generation

            elif task_type == 'class':
                def process_example(generation):
                    if eval_dataset == 'llm-aggrefact':
                        content = prompt.format(
                            document=generation['doc'],
                            claim=generation['claim']
                        )
                    elif 'info_bench' in eval_dataset:
                        content = prompt.format(
                            instruction=generation['instruction'],
                            response=generation['response'],
                            question=generation['question']
                        )

                    messages = [
                        {"role": "user", "content": content}
                    ]
                    if 'gair' in args.model.lower():
                        generation[f'text'] = LLAMA2_TEMPLATE.format(input=content)
                    else:
                        input_seq = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        generation[f'text'] = input_seq

                    return generation
            
            # Update dataset to get inputs for model
            updated_dataset = generation_dataset.map(process_example, num_proc=10) 

            if task_type == 'pair':
                # Run each data point twice (consistency)
                pair_inferences = []
                for idx in [1,2]:
                    prompts_in = updated_dataset[f"text_{idx}"]
                    inferences = model.generate(prompts_in, sampling_params)
                    outputs = [o.outputs[0].text for o in inferences]
                    pair_inferences.append(outputs)

                # Parse out the judgement from each output using get_prediction loaded from prompt_utils
                votes_1 = [get_prediction(inf, flip=False) for inf in pair_inferences[0]]
                votes_2 = [get_prediction(inf, flip=True) for inf in pair_inferences[1]]
                
                # Write all of the raw model outputs, inputs, parsed outputs (votes), and ground truth labels to a file
                with open(output_path, "w", buffering=1) as fw:
                    for i in range(len(votes_1)):
                        # Get the ground truth label
                        label_key = 'orig_preference' if eval_dataset == 'preference_bench' else 'label'
                        label = updated_dataset[i][label_key]

                        # Enforce that response A = 1, response B = 2, tie = 3 across all datasets
                        # For Auto_J dataset, need to add one to the label, raw data has response A = 0, response B = 1, tie = 2
                        if eval_dataset == 'auto_j':
                            label += 1
                        # Preference bench stores labels as 'A' or 'B'. Convert to ints.
                        elif eval_dataset == 'preference_bench':
                            label_new = 1 if label == 'A' else 2
                            label = label_new

                        # Store each sample
                        output = {
                            "votes": [votes_1[i], votes_2[i]],
                            "label": label,
                            "input_1": updated_dataset[i][f"text_1"],
                            "swap_inference1": pair_inferences[0][i],
                            "input_2": updated_dataset[i][f"text_2"],
                            "swap_inference2": pair_inferences[1][i],
                        }

                        fw.write(json.dumps(output)+"\n")

                # Compute metrics using the file we just saved
                output_evaluation = compute_acc(output_path)
            
            # Same process as above, don't need to run each sample twice for point (single rating) and classification
            elif task_type == 'point':
                prompts_in = updated_dataset['text']
                inferences = model.generate(prompts_in, sampling_params)
                outputs = [o.outputs[0].text for o in inferences]
                scores = [get_prediction(o) for o in outputs]

                with open(output_path, 'w', buffering=1) as fw:
                    for i in range(len(scores)):
                        output = {
                            "rating": scores[i],
                            "input" : updated_dataset[i]['text'],
                            "output": outputs[i],
                        }

                        # Some datasets have human scores and some datasets have GPT-scored outputs. We use either/both as ground truth labels.
                        if 'human_score' in updated_dataset[i]:
                            output['human_score'] = updated_dataset[i]['human_score']
                        if 'gpt4_score' in updated_dataset[i]:
                            output['gpt4_score'] = updated_dataset[i]['gpt4_score']
                    
                        fw.write(json.dumps(output)+"\n")
            
                output_evaluation = compute_pointwise_metrics(output_path)
            
            elif task_type == 'class':
                prompts_in = updated_dataset['text']
                inferences = model.generate(prompts_in, sampling_params)
                outputs = [o.outputs[0].text for o in inferences]
                labels_pred = [get_prediction(o) for o in outputs]

                with open(output_path, 'w', buffering=1) as fw:
                    for i in range(len(labels_pred)):
                        output = {
                            "label_pred": labels_pred[i],
                            "label": updated_dataset[i]['label'],
                            "input": updated_dataset[i]['text'],
                            "output": outputs[i],
                        }
                    
                        fw.write(json.dumps(output)+"\n")
                
                output_evaluation = compute_classification_metrics(output_path)


            for k, v in output_evaluation.items():
                print(f"{k}: {v}")
            
            with open(output_path.removesuffix(".jsonl")+".evaluation.json", "w") as fw:
                json.dump(output_evaluation, fw, indent=4, sort_keys=True)
    
    # shutdown vllm
    destroy_model_parallel()
    destroy_distributed_environment()
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="the directory for storing results")
    parser.add_argument("--model", type=str, help="model checkpoint or name")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_gpus", default=16, type=int)
    parser.add_argument("--eval_dataset", type=str, help="which eval dataset to run")
    parser.add_argument("--sfr_prompt_template", type=str, help="which of our prompt templates to use")

    # decoding strategy
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--num_sequences", default=1, type=int)

    args = parser.parse_args()

    main(args)