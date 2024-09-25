import json
import scipy.stats

VALID_PAIR_EVAL_DATASETS = ["rewardbench", "auto_j", "instrusum", "hhh", "preference_bench", "eval_bias_bench", "lfqa_eval"]
VALID_POINT_EVAL_DATASETS = ["flask", "mt_bench", "feedback_bench", "biggen_bench"]
VALID_CLASS_EVAL_DATASETS = ["llm-aggrefact", "info_bench_expert"]

def compute_acc(output_path):
    accuracy_swap1 = []
    accuracy_swap2 = []
    consistency_swap = []

    
    with open(output_path, 'r') as fr:
        evaluation_result = [json.loads(line) for line in fr.readlines()]

    for generation in evaluation_result:
        vote_1, vote_2 = generation["votes"]

         # first pairwise run
        if vote_1 == generation["label"]:
            accuracy_swap1.append(1.0)     
        else:
            accuracy_swap1.append(0)

        # second pairwise run
        if vote_2 == generation["label"]:
            accuracy_swap2.append(1.0) 
        else:
            accuracy_swap2.append(0)

        # Consistency computation: The two results should be the same regardless of order of responses
        if vote_1 == vote_2:
            consistency_swap.append(1.0)
        else:
            consistency_swap.append(0)
 
    swap1_accuracy = sum(accuracy_swap1) * 100. / len(accuracy_swap1)
    swap2_accuracy = sum(accuracy_swap2) * 100. / len(accuracy_swap2)
    average_accuracy = (swap1_accuracy + swap2_accuracy) / 2
    consistency = sum(consistency_swap) * 100. / len(consistency_swap)

    output_evaluation = {}
    output_evaluation["Accuracy_Swap1"] = swap1_accuracy 
    output_evaluation["Accuracy_Swap2"] = swap2_accuracy 
    output_evaluation["Average_Accuracy"] = average_accuracy 
    output_evaluation["Consistency"] = consistency

    return output_evaluation

def get_prediction_AB_to_num(generation, flip=False):
    # pred_lines = [line for line in generation.split("\n") if "Result" in line]
    # if len(pred_lines) == 0:
    #     return 0 
    # preference = pred_lines[0].removeprefix("**Result:**").removeprefix("**Result:").strip()
    preference = generation[-1]
    if preference == '.':
        preference = generation[-2]
    
    if preference not in ['A', 'B']:
        return 0
    if "A" == preference:
        if not flip:
            return 1
        else:
            return 2
    elif "B" == preference:
        if not flip:
            return 2
        else:
            return 1
    else:
        return 0

def get_prediction_tie(generation, flip=False):
    if generation is None:
        return -1 

    preference = generation[-1]
    if preference == '.':
        generation_splits = generation.split('.')
        generation_splits = [g for g in generation_splits if g != '']
        generation = generation_splits[-1]
        preference = generation[-1]
    
    if preference not in ['A', 'B'] and len(generation) >= 3 and generation[-3:].lower() == 'tie':
        prefernce = 'tie'
 
    if "A" == preference:
        if not flip:
            return 1
        else:
            return 2
    elif "B" == preference:
        if not flip:
            return 2
        else:
            return 1
    elif preference.lower() == 'tie':
        return 3
    else:
        return -1

def get_prediction_AB(generation, flip=False):
    preference = generation[-1]
    if preference == '.':
        preference = generation[-2]
    if preference not in ['A', 'B']:
        return 0
    if "A" == preference:
        if not flip:
            return 'A'
        else:
            return 'B'
    elif "B" == preference:
        if not flip:
            return 'B'
        else:
            return 'A'
    else:
        return 0




def compute_pointwise_metrics(output_path):
    with open(output_path, 'r') as fr:
        evaluation_result = [json.loads(line) for line in fr.readlines()]
    human_scores = []
    gpt4_scores = []
    generation_scores = []
    for example in evaluation_result:
        if example["rating"] is not None:
            generation_scores.append(example["rating"])
            if "human_score" in example:
                if isinstance(example["human_score"], list):
                    human_scores.append(sum(example["human_score"]) * 1.0 / len(example["human_score"]))
                else:
                    assert isinstance(example["human_score"], int)
                    human_scores.append(example["human_score"] * 1.0)
            if "gpt4_score" in example:
                if isinstance(example["gpt4_score"], list):
                    gpt4_scores.append(sum(example["gpt4_score"]) * 1.0 / len(example["gpt4_score"]))
                elif isinstance(example["gpt4_score"], float):
                    gpt4_scores.append(example["gpt4_score"])
                else:
                    assert isinstance(example["gpt4_score"], int)
                    gpt4_scores.append(example["gpt4_score"] * 1.0)

    output_evaluation = {}
    if len(human_scores) != 0:
        human_pearson = scipy.stats.pearsonr(generation_scores, human_scores)
        human_spearman = scipy.stats.spearmanr(generation_scores, human_scores)
        human_kendall = scipy.stats.kendalltau(generation_scores, human_scores)
        print("Human Pearson's r:\t\t", human_pearson)
        print("Human Spearman's rho:\t\t", human_spearman)
        print("Human Kendall's tau:\t\t", human_kendall)
        output_evaluation["human_pearson"] = human_pearson 
        output_evaluation["human_spearman"] = human_spearman 
        output_evaluation["human_kendall"] = human_kendall

    if len(gpt4_scores) != 0:
        gpt4_pearson = scipy.stats.pearsonr(generation_scores, gpt4_scores)
        gpt4_spearman = scipy.stats.spearmanr(generation_scores, gpt4_scores)
        gpt4_kendall = scipy.stats.kendalltau(generation_scores, gpt4_scores)
        print("GPT4 Pearson's r:\t\t", gpt4_pearson)
        print("GPT4 Spearman's rho:\t\t", gpt4_spearman)
        print("GPT4 Kendall's tau:\t\t", gpt4_kendall)
        output_evaluation["gpt4_pearson"] = gpt4_pearson
        output_evaluation["gpt4_spearman"] = gpt4_spearman
        output_evaluation["gpt4_kendall"] = gpt4_kendall

    return output_evaluation

def compute_classification_metrics(output_path):
    accuracy = []
    
    with open(output_path, 'r') as fr:
        evaluation_result = [json.loads(line) for line in fr.readlines()]

    for generation in evaluation_result:
        label_pred, label = generation['label_pred'], generation['label']
       
        if label_pred == label:
            accuracy.append(1.0)
        else:
            accuracy.append(0)
 
    overall_acc = sum(accuracy) * 100. / len(accuracy)
    
    output_evaluation = {}
    output_evaluation["accuracy"] = overall_acc 

    return output_evaluation