import importlib
from utils import get_prediction_AB, get_prediction_AB_to_num, get_prediction_tie

def get_prompt(eval_dataset, model_modifier):

    # autoj has ties in eval, load special prompts to accommodate
    if eval_dataset == "auto_j":
        # Default prompt for auto_j is their dataset's prompt. Load that for instruct models and Auto-J model
        if model_modifier in ['', 'auto_j']:
            from prompts.auto_j import PROMPT_PAIRWISE_TIE as prompt
            from prompts.auto_j import get_prediction_tie as get_prediction
            prompt_name = 'auto_j_default_pairwise'

        # sfr-judges prompts
        elif model_modifier[:3] == 'sfr':
            get_prediction = importlib.import_module('prompts.sfr').get_prediction_tie

            # task-specific
            if 'task_specific' in model_modifier:
                from prompts.sfr import PROMPT_PAIRWISE_AUTOJ as prompt
                prompt_name = f'{model_modifier}_pairwise'
            # reward bench
            elif 'rewardbench' in model_modifier:
                from prompts.sfr import PROMPT_PAIRWISE_TIE_RB as prompt
                prompt_name = f'{model_modifier}_pairwise'
            # prepair
            elif 'prepair' in model_modifier:
                from prompts.sfr_prepair import PROMPT_PAIRWISE_TIE as prompt
                prompt_name = f'{model_modifier}_pairwise'

        # other judge models, use judge-specific prompts
        else:
            prompt = importlib.import_module('prompts.{}'.format(model_modifier)).PROMPT_PAIRWISE_TIE
            prompt_name = f"{model_modifier}_pairwise_tie"
            get_prediction = importlib.import_module('prompts.{}'.format(model_modifier)).get_prediction_tie

    # preference bench has rubrics and reference answers, load prompts
    elif eval_dataset == "preference_bench":
        if model_modifier in ['', 'prometheus']:
            from prompts.prometheus import PROMPT_PAIRWISE_RUBRIC_REF as prompt
            from prompts.prometheus import get_prediction as get_prediction
            prompt_name = 'prometheus_rubric_ref_default_pairwise'
        elif model_modifier[:3] == 'sfr':
            get_prediction = importlib.import_module('prompts.sfr').get_prediction_tie

            # task-specific
            if 'task_specific' in model_modifier:
                from prompts.sfr import PROMPT_PAIRWISE_RUBRIC_REF as prompt
                prompt_name = f'{model_modifier}_pairwise_rubric_ref'
            # reward bench
            elif 'rewardbench' in model_modifier:
                from prompts.sfr import PROMPT_PAIRWISE_RUBRIC_REF_RB as prompt
                prompt_name = f'{model_modifier}_pairwise_rubric_ref'
            # prepair
            elif 'prepair' in model_modifier:
                from prompts.sfr_prepair import PROMPT_PAIRWISE_RUBRIC_REF as prompt
                prompt_name = f'{model_modifier}_pairwise_rubric_ref'
    
        else:
            prompt = importlib.import_module('prompts.{}'.format(model_modifier)).PROMPT_PAIRWISE_RUBRIC_REF
            prompt_name = f"{model_modifier}_pairwise_rubric_ref"
            get_prediction = importlib.import_module('prompts.{}'.format(model_modifier)).get_prediction

    # if not the above two datasets, but is SFR-judges model, load task-specific dataset if necessary
    elif model_modifier[:3] == 'sfr':
        if 'prepair' in model_modifier:            
            from prompts.sfr_prepair import PROMPT_PAIRWISE as prompt
            from prompts.sfr import get_prediction as get_prediction
            prompt_name = f'{model_modifier}_pairwise'
        elif 'rewardbench' in model_modifier:
            from prompts.sfr import PROMPT_PAIRWISE_RB as prompt
            from prompts.sfr import get_prediction as get_prediction
            prompt_name = f'{model_modifier}_pairwise'
        elif 'task_specific' in model_modifier:
            if eval_dataset == 'lfqa_eval':
                from prompts.sfr import PROMPT_PAIRWISE_LFQA as prompt                
            elif eval_dataset == 'hhh':
                from prompts.sfr import PROMPT_PAIRWISE_HHH as prompt
            elif eval_dataset == 'instrusum':
                from prompts.sfr import PROMPT_PAIRWISE_INSTRUSUM as prompt
            elif eval_dataset == 'eval_bias_bench':
                from prompts.sfr import PROMPT_PAIRWISE_EVALBIASBENCH as prompt
                
            prompt_name = f"{model_modifier}_pairwise_{eval_dataset}"
            get_prediction = importlib.import_module('prompts.sfr').get_prediction
    
    # Not SFR-Judge or Auto-J/PreferenceBench datasets
    else:
        # Use default prompts if no judge-specific prompt included
        if model_modifier == '':
            from prompts.default import PROMPT_PAIRWISE as prompt
            from prompts.default import get_prediction as get_prediction
            prompt_name = 'default_pairwise'
        # Use judge-specific prompt
        else:
            prompt = importlib.import_module('prompts.{}'.format(model_modifier)).PROMPT_PAIRWISE
            prompt_name = f"{model_modifier}_pairwise"
            get_prediction = importlib.import_module('prompts.{}'.format(model_modifier)).get_prediction

    return prompt, prompt_name, get_prediction


# One big parsing function for pointwise evaluation
def get_prediction_pointwise(inference):
    result_line = [line for line in inference.split("\n") if any(s in line for s in ["Result", "Score", "[RESULT]", "Output", "[[", "]]"])]
    score = 0.0

    if len(result_line) != 0:
        score_span = ""
        if "**Score:**" in result_line[0]:
            score_span = result_line[0].removeprefix("**Score:**").removeprefix("**Score:").strip()
        elif 'Score:' in result_line[0]:
            score_span = result_line[0].removeprefix("Score:").strip().removeprefix("Score:").strip()
        elif 'Score' in result_line[0]:
            score_span = result_line[0].removeprefix("Score").strip().removeprefix("Score").strip()
        elif 'Output:' in result_line[0]:
            score_span = result_line[0].removeprefix("Output:").strip().removeprefix("Output:").strip()
        elif 'Output' in result_line[0]:
            score_span = result_line[0].removeprefix("Output").strip().removeprefix("Output").strip()
        elif "**Result:**" in result_line[0]:
            score_span = result_line[0].removeprefix("**Result:**").removeprefix("**Result:").strip()
        elif "Result:" in result_line[0]:
            score_span = result_line[0].removeprefix("Result:").strip().removeprefix("Result:").strip()
        elif "Result" in result_line[0]:
            score_span = result_line[0].removeprefix("Result").strip().removeprefix("Result").strip()
        elif "[Result]" in result_line[0]:
            score_span = result_line[0].removeprefix("[Result]").removeprefix("[Result]").strip()
        elif "[RESULT]" in result_line[0]:
            score_span = result_line[0].removeprefix("[RESULT]").removeprefix("[RESULT]").strip()
            
        if score_span != "" and score_span[0].isdigit():
            score = float(score_span[0])

    # If score is 0.0 but inference is non-empty, check last entry. If it's convertable to float, then return that as score
    # All prompts ask the model to output the score last, so this is a catch-all
    if score == 0.0 and inference.strip() != '' and inference.strip()[-1].isdigit():
        try:
            score = float(inference.strip()[-1])
        except:
            # Ultra rare case when the produced score is in superscript (occurred for Mistral-7b-Instruct-v0.1)
            # superscript(2).isdigit() => True, but cannot be converted to float
            # Conversion to normal number from: https://codegolf.stackexchange.com/questions/238462/convert-superscript-numbers-to-normal-numbers
            try:
                score = float(inference.strip()[-1].translate('%7d   45678923'%10*999))
                assert score in [1., 2., 3., 4., 5.]
            except:
                score = 0.0

    return score

    
#['flask', 'mt_bench', 'feedback_bench', 'biggen_bench']
def get_prompt_point(eval_dataset, model_modifier):
    get_prediction = None
    if model_modifier == '' or model_modifier == 'prometheus':
        from prompts.default import PROMPT_ABSOLUTE_RUBRIC_REF as prompt
        prompt_name = 'default_pointwise'
        get_prediction = get_prediction_pointwise
    #No task-specific considerations for SFR-Judges, since each of these eval sets has rubrics/reference answers
    elif model_modifier[:3] == 'sfr':
        from prompts.sfr import PROMPT_ABSOLUTE_RUBRIC_REF as prompt
        prompt_name = f'sfr_pointwise_rubric_ref'
        get_prediction = get_prediction_pointwise
    elif model_modifier in ['auto_j', 'skywork']: # need custom prediction parsing
        prompt = importlib.import_module('prompts.{}'.format(model_modifier)).PROMPT_ABSOLUTE_RUBRIC_REF
        prompt_name = f"{model_modifier}_rubric_ref"
        get_prediction = importlib.import_module('prompts.{}'.format(model_modifier)).extract_single_rating
    else:
        prompt = importlib.import_module('prompts.{}'.format(model_modifier)).PROMPT_ABSOLUTE_RUBRIC_REF
        prompt_name = f"{model_modifier}_rubric_ref"
        get_prediction = get_prediction_pointwise
        
    return prompt, prompt_name, get_prediction


def get_prediction_classification(inference):
    if "**Result:**" in inference:
        result = inference.split('**Result:**')[-1].strip().lower()
        if '.' in result:
            result = result.split('.')[0].strip().lower()
        if result == 'yes':
            return 1
        elif result == 'no':
            return 0
        else:
            return None
    else:
        inference = inference.lower()
        if inference == 'yes' or ('yes' in inference and 'no' not in inference):
            return 1
        elif inference == 'no' or ('no' in inference and 'yes' not in inference):
            return 0
        else:
            return None

def get_prompt_classification(eval_dataset, model_modifier):
    if eval_dataset == 'llm-aggrefact':
        if model_modifier[:3] == 'sfr':
            from prompts.sfr import PROMPT_AGGREFACT as prompt
            prompt_name = 'sfr_classification_aggrefact'
        else:
            from prompts.aggrefact_prompt import AGGREFACT_PROMPT as prompt
            prompt_name = 'default_classification'
    elif 'info_bench' in eval_dataset:
        if model_modifier[:3] == 'sfr':
            from prompts.sfr import PROMPT_INFOBENCH as prompt
            prompt_name = 'sfr_classification_infobench'
        else:
            from prompts.infobench_prompt import INFOBENCH_PROMPT as prompt
            prompt_name = 'default_classification_infobench'

    get_prediction = get_prediction_classification
    return prompt, prompt_name, get_prediction