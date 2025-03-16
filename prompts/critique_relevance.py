system_prompt = """
    Please evaluate the relevance of the answer to the question provided below.

    Question: "{question}"

    Answer: "{answer}"

    Critique: Based on the question and answer, determine how relevant the answer is to the question. Provide an integer score from 1 (not relevant) to 5 (highly relevant) and a brief explanation justifying your score.
    Return your result in the following format: Relevance: [score] Explanation: [explanation].

    An example is:

    Relevance: 3 Explanation: The answer is somewhat relevant to the question, but it could be improved by providing more context.
    """

user_prompt = ""