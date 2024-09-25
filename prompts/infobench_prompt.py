INFOBENCH_PROMPT="""
Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:
- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration, consider a question that asks, “Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be ‘YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question.
- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks, “Is the second sentence in the generated text a compound sentence?” and the generated text only has one sentence, it offers no relevant information to answer the question. Consequently, the answer should be ‘NO'.

Before providing your answer, you should first provide reasoning for your response.

Your reply should strictly follow this format:
**Reasoning:** <feedback evaluating the documant and claim>

**Result:** <Yes or No>

Input:
{instruction}

Generated Text:
{response}

Question:
{question}
""".strip()