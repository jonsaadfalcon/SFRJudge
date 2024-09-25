AGGREFACT_PROMPT="""
Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that
all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

Please assess the claim’s consistency with the document by responding first with reasoning then concluding with either "yes" or "no". 

Your reply should strictly follow this format:
**Reasoning:** <feedback evaluating the documant and claim>

**Result:** <Yes or No>

Document: {document}

Claim: {claim}

Answer:
"""