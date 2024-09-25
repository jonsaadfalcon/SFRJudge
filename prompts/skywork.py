# Prompts sourced from: https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-8B
# Minimal changes are made to accommodate new info (e.g., rubrics and references)

PROMPT_PAIRWISE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{input}

[The Start of Assistant A's Answer]
{output_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{output_2}
[The End of Assistant B's Answer]
"""

PROMPT_PAIRWISE_TIE= """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, "[[C]]" if tie, where the answers are the same quality.

[User Question]
{input}

[The Start of Assistant A's Answer]
{output_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{output_2}
[The End of Assistant B's Answer]
"""

PROMPT_PAIRWISE_RUBRIC_REF= """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the scoring rubric better. 
Your evaluation should consider the factors outlined in the rubric. Use the reference answer as a guide for your decision. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{input}

[The Start of Assistant A's Answer]
{output_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{output_2}
[The End of Assistant B's Answer]

[The Start of the Rubric]
{rubric}
[The End of the Rubric]

[The Start of the Reference Answer]
{reference_answer}
[The End of the Reference Answer]
"""

PROMPT_ABSOLUTE_RUBRIC_REF="""Please act as an impartial judge and evaluate the quality of a response provided by an AI assistant to the user question displayed below. You should choose the score that follows the scoring rubric better. 
Your evaluation should consider the factors outlined in the rubric. Use the reference answer as a guide for your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[an integer between 1 and 5]]".

[User Question]
{instruction}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]

[The Start of the Rubric]
{rubric}
[The End of the Rubric]

[The Start of the Reference Answer]
{reference_answer}
[The End of the Reference Answer]
"""

def get_prediction(generation, flip=False):
    if "[[A]]" in generation:
        if not flip:
            return 1
        else:
            return 2
    elif "[[B]]" in generation:
        if not flip:
            return 2
        else:
            return 1
    else:
        return -1

def get_prediction_tie(generation, flip=False):
    if "[[A]]" in generation:
        if not flip:
            return 1
        else:
            return 2
    elif "[[B]]" in generation:
        if not flip:
            return 2
        else:
            return 1
    elif "[[C]]" in generation:
        return 3
    else:
        return -1

# Auto-J parsing code, they use same format as Auto-J
def extract_single_rating(score_output):
    if "[[" in score_output:
        pos = score_output.rfind("[[")
        pos2 = score_output.find("]]", pos)

        try:
            assert pos != -1 and pos2 != -1
            return float(score_output[pos + len("[["):pos2].strip())
        except:
            return 0.0
    else:
        return 0.0