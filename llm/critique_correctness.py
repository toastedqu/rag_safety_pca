import pandas as pd
from tqdm import tqdm

from openai import OpenAI


def output_llm(required_files, system_prompt, user_prompt, model_name):
    """
    Output the LLM code snippet for rephrasing queries
    :param required_files: List[str] - list of required files
    :param system_prompt: str - system prompt
    :param user_prompt: str - user prompt
    :return: str
    """
    x = pd.read_csv(required_files[0])

    client = OpenAI()

    answers = x["response"].tolist()

    responses = []

    for r in tqdm(answers, total=len(answers)):
        new_prompt = system_prompt.replace("{answer}", r)

        messages = [
            {"role": "user", "content": new_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )

        response = response.choices[0].message.content

        try:
            score = int(response.split("Correctness: ")[1].split(" Explanation")[0])
        except:
            try:
                score = int(response.split("Correctness: ")[1].split("\nExplanation")[0])
            except:
                try:
                    score = int(response.split("Correctness: ")[1].split(".")[0])
                except:
                    score = response

        responses.append(score)

    x["correctness_score"] = responses

    return x
