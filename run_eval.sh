#!/bin/bash
set -e

num_gpus=1 
export CUDA_VISIBLE_DEVICES=0

# sampling parameters
temperature=0
top_p=1.0


### Run using model hosted on Huggingface and store results locally 

# Set directory to store outputs
model="<HUGGINGFACE MODEL REPO HERE>"
eval_dir="./models/${model}/eval_result"
output_path="${eval_dir}/{dataset_name}/judge_{prompt}_{signature}.jsonl" # main_eval.py will format `prompt` and `signature`

python -u \
    main_eval.py \
    --model $model \
    --num_gpus $num_gpus \
    --eval_dataset all \
    --output_path $output_path \
    --temperature $temperature \
    --top_p $top_p \



### Run using local model hosted on Huggingface and store results in local model folder 
model=/path/to/your/local/model
eval_dir="${model}/eval_result"
output_path="${eval_dir}/{dataset_name}/judge_{prompt}_{signature}.jsonl"

python -u \
    main_eval.py \
    --model $model \
    --num_gpus $num_gpus \
    --eval_dataset all \
    --output_path $output_path \
    --temperature $temperature \
    --top_p $top_p \