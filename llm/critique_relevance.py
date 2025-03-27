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

    queries = x["rephrased_query"].tolist()
    answers = x["response"].tolist()

    responses = []

    for q, r in tqdm(zip(queries, answers), total=len(queries)):
        system_prompt = system_prompt.replace("{question}", q).replace("{answer}", r)

        messages = [
            {"role": "user", "content": system_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )

        response = response.choices[0].message.content

        try:
            score = int(response.split("Relevance: ")[1].split(" Explanation")[0])
        except:
            try:
                score = int(response.split("Relevance: ")[1].split("\nExplanation")[0])
            except:
                try:
                    score = int(response.split("Relevance: ")[1].split(".")[0])
                except:
                    score = response

        responses.append(score)
        print(q)
        print(r)
        print(score)

    x["relevance_score"] = responses

    return x
