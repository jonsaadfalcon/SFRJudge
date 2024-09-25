import os
import glob
import json
import argparse
import numpy as np
from collections import defaultdict
from utils import VALID_PAIR_EVAL_DATASETS, VALID_POINT_EVAL_DATASETS, VALID_CLASS_EVAL_DATASETS

def read_jsonl(file_path):
    with open(file_path, 'r') as fr:
        data = [json.loads(line) for line in fr.readlines()]
    return data

def read_json(file_path):
    with open(file_path, 'r') as fr:
        data = json.load(fr)
    return data

def write_jsonl_raw(data, jfile):
    with open(jfile, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

def write_json(data, file_path, sort=False):
    if sort:
        data = {key:data[key] for key in sorted(data.keys())}

    with open(file_path, "w+") as f:  # save format for AI2 beaker to show results
        json.dump(data, f, indent=2)


# Aggregate saved eval folders into on file that summarizes performance for pairwise datasets
def process_pairwise(args):
    aggregated_outputs = {}


    for ds in VALID_PAIR_EVAL_DATASETS:
        print(f"EVALUATING: {ds}")
        ds_eval = {}
        eval_path = os.path.join(args.eval_path, ds)

        if ds == 'rewardbench':
            eval_path = os.path.join(args.eval_path, 'rewardbench/scores.json')
            if not os.path.exists(eval_path):
                continue

            eval_results = read_json(eval_path)
            ds_eval = eval_results['leaderboard']
            aggregated_outputs[ds] = ds_eval
            continue

        # get dataset file

        data_files = list(glob.glob(f"{eval_path}/*.evaluation.json"))
        print(eval_path)
        if len(data_files) == 1:
            eval_results = read_json(os.path.join(eval_path, data_files[0]))
            if args.acc_type == 'best':
                ds_eval['overall'] = max(eval_results['Accuracy_Swap1'], eval_results['Accuracy_Swap2'])
            elif args.acc_type == 'pos1':
                ds_eval['overall'] = eval_results['Accuracy_Swap1']
            elif args.acc_type == 'pos2':
                ds_eval['overall'] = eval_results['Accuracy_Swap1']
            elif args.acc_type == 'avg':
                ds_eval['overall'] = eval_results['Average_Accuracy']

            ds_eval['avg_consistency'] = eval_results['Consistency'] # not really an avg, but keep keys consistent
        else:
            ds_eval['splits'] = {}
            vals = []
            con_vals = []
            weights = []

            # compute microaverage for datasets with subsets
            if ds == 'eval_bias_bench':
                split_to_score = {
                    'biasbench_content_continuation': 12,
                    'biasbench_concreteness': 14,
                    'biasbench_nested_instruction': 12,
                    'biasbench_familiar_knowledge_preference_bias': 12,
                    'biasbench_empty_reference': 13,
                    'biasbench_length_bias': 17,
                }
            elif ds == 'hhh':
                split_to_score = {
                    "hhh_other": 43,
                    "hhh_honest": 61,
                    "hhh_harmless": 58,
                    "hhh_helpful": 59,
                }
            else:
                split_to_score = defaultdict(lambda: 1)

            for df in data_files:
                split_name = df.split('/')[-1].split('-')[0]
                print(split_name)
                
                eval_results = read_json(os.path.join(eval_path, df))
                if args.acc_type == 'best':
                    ds_eval['splits'][split_name] = max(eval_results['Accuracy_Swap1'], eval_results['Accuracy_Swap2'])
                elif args.acc_type == 'pos1':
                    ds_eval['splits'][split_name] = eval_results['Accuracy_Swap1']
                elif args.acc_type == 'pos2':
                    ds_eval['splits'][split_name] = eval_results['Accuracy_Swap1']
                elif args.acc_type == 'avg':
                    ds_eval['splits'][split_name] = eval_results['Average_Accuracy']
                ds_eval['splits'][f"{split_name}-consistency"] = eval_results['Consistency']

                vals.append(ds_eval['splits'][split_name])
                con_vals.append(ds_eval['splits'][f"{split_name}-consistency"])
                weights.append(split_to_score[split_name])
            
            ds_eval['overall'] = np.average(vals, weights=weights) #TODO weighted avg based on num samples for HHH and Biasbench
            ds_eval['avg_consistency'] = np.average(con_vals, weights=weights)

        aggregated_outputs[ds] = ds_eval

    save_path = os.path.join(args.eval_path, f"pairwise_eval_results_acc_{args.acc_type}.jsonl")
    write_json(aggregated_outputs, save_path)


# Aggregate saved eval folders into on file that summarizes performance for single rating and classification datasets
def process_pointwise(args):
    aggregated_outputs = {}

    datasets = [x for x in VALID_POINT_EVAL_DATASETS]
    if args.type == 'point' or args.type == 'all':
        datasets += [x for x in VALID_CLASS_EVAL_DATASETS]

    for ds in datasets:
        print(f"EVALUATING: {ds}")
        ds_eval = {}
        eval_path = os.path.join(args.eval_path, ds)

        # get dataset file
        data_files = list(glob.glob(f"{eval_path}/*.evaluation.json"))
        if len(data_files) == 1:
            eval_results = read_json(os.path.join(eval_path, data_files[0]))
            ds_eval['overall'] = eval_results
        else:
            ds_eval['splits'] = {}
            h_pearson = []
            gpt_pearson = []
            for df in data_files:
                split_name = df.split('/')[-1].split('-')[0]

                # We run biggen_bench subset by subset and by aggregating the entire dataset
                # Makes computing an overall pearson coeff. easier
                if ds == 'biggen_bench' and split_name != 'human_eval':
                    continue
                
                eval_results = read_json(os.path.join(eval_path, df))
                ds_eval['splits'][split_name] = eval_results
                
                if 'human_pearson' in ds_eval['splits'][split_name]:
                    h_pearson.append(ds_eval['splits'][split_name]['human_pearson'][0])
                if 'gpt4_pearson' in ds_eval['splits'][split_name]:
                    gpt_pearson.append(ds_eval['splits'][split_name]['gpt4_pearson'][0])
            
            overall_scores = {}
            if h_pearson is not None:
                overall_scores['human_pearson'] = np.average(h_pearson)
            if gpt_pearson is not None:
                overall_scores['gpt4_pearson'] = np.average(gpt_pearson)
            ds_eval['overall'] = overall_scores

        aggregated_outputs[ds] = ds_eval
        

    save_path = os.path.join(args.eval_path, f"pointwise_eval_results_acc_{args.acc_type}.jsonl")
    write_json(aggregated_outputs, save_path)

# Takes outputs of process_pairwise() and process_pointwise() and further aggregates into a "leaderboard"
# Outputs here used for our paper tables
def compile_leaderboard(args):
    agg_files = []
    if args.type == 'pair' or args.type == 'all':
        agg_path = os.path.join(args.eval_path, f"pairwise_eval_results_acc_{args.acc_type}.jsonl")
        agg_files.append(agg_path)
    if 'point' in args.type or args.type == 'all':
        agg_path = os.path.join(args.eval_path, f"pointwise_eval_results_acc_{args.acc_type}.jsonl")
        agg_files.append(agg_path)

    
    leaderboard = {}
    for agg_path in agg_files:
        agg_result = read_json(agg_path)
        for k, v in agg_result.items():
            store_key = ''
            if k in VALID_PAIR_EVAL_DATASETS:
                store_key = f'pairwise-{k}'
            elif k in VALID_POINT_EVAL_DATASETS:
                store_key = f'pointwise-{k}'
            elif k in VALID_CLASS_EVAL_DATASETS:
                store_key = f'classification-{k}'

            if k == 'rewardbench':
                lb_vals = {subset:100*score for subset, score in v.items()} # assumes score is in [0,1]
                leaderboard[store_key] = lb_vals
            else:
                eval_results = v['overall']
                if isinstance(eval_results, dict):
                    lb_vals = {metric: eval_results[metric] for metric in eval_results.keys() if metric in ['accuracy', 'gpt4_pearson', 'human_pearson']}
                else:
                    lb_vals = v
                    lb_vals['accuracy'] = lb_vals.pop('overall')
                    if 'splits' in lb_vals:
                        lb_vals.pop('splits')
                leaderboard[store_key] = lb_vals

        if 'pairwise' in agg_path:
            scores = []
            consist = []
            for dsname in VALID_PAIR_EVAL_DATASETS:
                ds = f'pairwise-{dsname}'
                if ds in leaderboard:
                    if dsname == 'rewardbench':
                        scores.append(leaderboard[ds]['overall_score'])
                    else:
                        scores.append(leaderboard[ds]['accuracy'])
                        consist.append(leaderboard[ds]['avg_consistency'])

            leaderboard['average_pairwise'] = np.average(scores)
            leaderboard['average_pairwise_consistency'] = np.average(consist)

        if 'pointwise' in agg_path:
            scores = []
            for dsname in VALID_POINT_EVAL_DATASETS: #+ VALID_CLASS_EVAL_DATASETS:
                ds = f'pointwise-{dsname}'

                if 'gpt4_pearson' in leaderboard[ds]:
                    if dsname == 'biggen_bench':
                        scores.append(leaderboard[ds]['gpt4_pearson'])
                    else:
                        scores.append(leaderboard[ds]['gpt4_pearson'][0])

                if 'human_pearson' in leaderboard[ds]:
                    if dsname == 'biggen_bench':
                        scores.append(leaderboard[ds]['human_pearson'])
                    else:
                        scores.append(leaderboard[ds]['human_pearson'][0])
                print(ds)
                print(scores)

            leaderboard['average_pointwise'] = np.average(scores)
    
    save_path = os.path.join(args.eval_path, f'leaderboard_results.jsonl')
    write_json(leaderboard, save_path, sort=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, help="the folder where results are stored ([model]/eval_result/)")
    parser.add_argument("--acc_type", type=str, default='best', choices = ["pos1", "pos2", "avg", "best"], help="accuracy for pairwise")
    parser.add_argument("--type", type=str, default='all', choices=["pair", "point", "point_no_class", "all"])
    # --type: aggregate results for all datasets, pairwise only, pointwise (direct scoring and classification), direct scoring only.

    args = parser.parse_args()

    if args.type == 'pair' or args.type == 'all':
        process_pairwise(args)
    if 'point' in args.type or args.type == 'all':
        process_pointwise(args)

    compile_leaderboard(args)

