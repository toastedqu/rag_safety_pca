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
    try:
        kb = pd.read_csv(required_files[0])
    except:
        kb = pd.read_csv(required_files[0], encoding="unicode_escape")

    kb = kb["user_kp"].tolist()

    good_kb = []

    for q in kb:
        if f"Query: {q}" not in good_kb:
            good_kb.append(f"Query: {q}")

    good_kb = "\n".join(good_kb)

    kb = pd.read_csv(required_files[1])["Query"].tolist()

    bad_kb = []

    for q in kb:
        if f"Query: {q}" not in bad_kb:
            bad_kb.append(f"Query: {q}")

    bad_kb = "\n".join(bad_kb)

    queries_to_eval = pd.read_csv(required_files[2])["rephrased_query"].tolist()

    client = OpenAI()

    responses = []

    for q in tqdm(queries_to_eval):
        system_prompt = system_prompt.replace("{kb}", good_kb).replace("{adversarial_kb}", bad_kb).replace("{question}", q)

        messages = [
            {"role": "user", "content": system_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )

        responses.append(response.choices[0].message.content)

    df_file = pd.DataFrame({"query": queries_to_eval, "decision": responses})

    return df_file
