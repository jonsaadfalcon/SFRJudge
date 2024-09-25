# Direct scoring/Pointwise eval: AutoJ originally prompts models to score on a scale of 1-10. We change this to be 1-5, in alignment with benchmarks
# Prompts sourced from: https://github.com/GAIR-NLP/auto-j/blob/main/codes/usage/constants_prompt.py
# Minimal changes are made to accommodate new info (e.g., rubrics and references)

PROMPT_ABSOLUTE = """
Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {instruction}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 5 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
""".strip()

PROMPT_ABSOLUTE_RUBRIC_REF="""
Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {instruction}
***
[Response]: {response}
***
[Rubric]: {rubric}
***
[Reference answer]: {reference_answer}
***
[END DATA]

Write critiques for this response based on the provided rubric and reference answer. After that, you should give a final rating for the response on a scale of 1 to 5 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
""".strip()

PROMPT_PAIRWISE = """
You are assessing two submitted responses on a given user's query and judging which response is better. Here is the data:

[BEGIN DATA]
***
[Query]: {input}
***
[Response 1]: {output_1}
***
[Response 2]: {output_2}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better. Begin your final decision statement with "So, the final decision is Response 1 / Response 2". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.
""".strip()

PROMPT_PAIRWISE_TIE = """
You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {input}
***
[Response 1]: {output_1}
***
[Response 2]: {output_2}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.
""".strip()

PROMPT_PAIRWISE_RUBRIC_REF = """
You are assessing two submitted responses on a given user's query and judging which response is better. Here is the data:

[BEGIN DATA]
***
[Query]: {input}
***
[Response 1]: {output_1}
***
[Response 2]: {output_2}
***
[Rubric]: {rubric}
***
[Reference answer]: {reference_answer}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses. Use the provided rubric and reference answer as guides in your evaluation.
2. Conclude your comparison by providing a final decision on which response is better. Begin your final decision statement with "So, the final decision is Response 1 / Response 2". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.
""".strip()

def get_prediction_tie(raw_output, flip=False):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if 'response 1' in pred_rest.lower():#pred_rest.startswith('response 1'):
            if flip:
                pred_label = 2
            else:
                pred_label = 1
        elif 'response 2' in pred_rest.lower(): #pred_rest.startswith('response 2'):
            if flip:
                pred_label = 1
            else:
                pred_label = 2
        elif 'tie' in pred_rest.lower(): #pred_rest.startswith('tie'):
            pred_label = 3
    return pred_label


def get_prediction(raw_output, flip=False):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            if flip:
                pred_label = 2
            else:
                pred_label = 1
        elif pred_rest.startswith('response 2'):
            if flip:
                pred_label = 1
            else:
                pred_label = 2
    return pred_label

# Parsing code sourced from: https://github.com/GAIR-NLP/auto-j/blob/2ae17a3965d933232e9cd50302aa0f176249c83b/codes/usage/example.py#L21
def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)

        try:
            assert pos != -1 and pos2 != -1
            return float(score_output[pos + len("Rating: [["):pos2].strip())
        except:
            print(f"score_output: {score_output} | pos: {pos}, {pos2}")
            return 0.0
    else:
        return 0.0
