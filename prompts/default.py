# default prompt from rewardbench
PROMPT_PAIRWISE="""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You
should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not
influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be
as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is
better and "[[B]]" if assistant B is better.

[User Question]
{input}
[The Start of Assistant A’s Answer]
{output_1}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{output_2}
[The End of Assistant B’s Answer]
""".strip()


# Default prompt from Prometheus pointwise eval
PROMPT_ABSOLUTE_RUBRIC_REF = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

Your reply should only contain the following:

**Reasoning:** <Your feedback>

**Result:** <an integer between 1 and 5>


###Feedback: """.strip()

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