import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from sklearn.utils import shuffle
import random


def output_llm(required_files, system_prompt, user_prompt, model_name):
    """
    Output the LLM code snippet for rephrasing queries
    :param required_files: List[str] - list of required files
    :param system_prompt: str - system prompt
    :param user_prompt: str - user prompt
    :return: str
    """
    x = pd.read_csv(required_files[0])
    x = x[x["secondary intent"] != "Covid"]

    client = OpenAI()

    questions = x["question"].tolist()
    questions = shuffle(questions)

    responses = []

    for q in tqdm(questions):
        type_of_re = random.choice(["question", "command-style statement"])

        system_prompt.replace("{query}", q).replace("{type_of_re}", type_of_re)

        messages = [
            {"role": "user", "content": system_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )

        responses.append(response.choices[0].message.content)

    df_file = pd.DataFrame({"original_query": questions, "rephrased_query": responses})

    return df_file
