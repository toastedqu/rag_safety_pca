import json
from typing import List, Tuple

from datasets import load_dataset
import pandas as pd


def load_local(id: str) -> List[pd.DataFrame]:
    """load COVID and Substance data

    Returns:
        (list(df)): a list of 2 dfs (COVID, Substance)
    """
    if id == "covid":
        return [
            pd.read_csv(r"data\response_db.csv", encoding="unicode_escape")
            .drop_duplicates(["user_kp", "system_kp"])
            .reset_index(drop=True)
        ]

    if id == "drugs":
        return [pd.read_csv(r"data\Substance_Use_and_Recovery_FAQ.csv")]

    raise ValueError("Invalid dataset id")


def json_to_df(path) -> pd.DataFrame:
    """convert given MSMARCO json dataset to df

    Args:
        path (str): path to each json file

    Returns:
        (df): dataframe per MSMARCO dataset
    """
    d = json.load(open(path, "r"))["data"]
    docs = [item["paragraphs"][0]["context"] for item in d]
    queries = [item["paragraphs"][0]["qas"][0]["question"] for item in d]
    return pd.DataFrame({"query": queries, "doc": docs})


def load_msmarco(id: str) -> pd.DataFrame:
    """load given MSMARCO dataset into df

    Args:
        id (str): domain name per stackexchange dataset

    Returns:
        (df): dataframe per stackexchange dataset
    """
    return pd.concat(
        [
            json_to_df(rf"data\msmarco\squad.{id}.train.json"),
            json_to_df(rf"data\msmarco\squad.{id}.dev.json"),
            json_to_df(rf"data\msmarco\squad.{id}.test.json"),
        ]
    ).reset_index(drop=True)


def load_stackexchange(id: str):
    """load given stackexchange dataset into df

    Args:
        id (str): domain name per stackexchange dataset

    Returns:
        (df): dataframe per stackexchange dataset
    """
    return load_dataset(
        "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
        id,
        split="train",
    )


def load_queries_docs(
    dataset: str, tags: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    load queries and docs for given dataset and tags
    :param dataset: str
    :param tags: str
    :return:
        queries: List[str] - list of queries
        docs: List[str] - list of docs
        tags: List[str] - list of tags
    """
    if dataset == "local":
        if tags == "all":
            tags = ["covid", "drugs"]
        else:
            tags = tags.split(",")

        queries = []
        docs = []
        for tag in tags:
            dfs = load_local(tag)

            if tag == "covid":
                queries.append(dfs[0]["user_kp"].tolist())
                docs.append(dfs[0]["system_kp"].tolist())

            if tag == "drugs":
                queries.append(dfs[1]["question"].tolist())
                docs.append(dfs[1]["relevant answer"].tolist())

        return queries, docs, tags

    if dataset == "msmarco":
        if tags == "all":
            tags = ["biomedical", "music", "film", "finance", "law", "computing"]
        else:
            tags = tags.split(",")

        dfs = [load_msmarco(tag) for tag in tags]
        queries = [df["query"].tolist() for df in dfs]
        docs = [df["doc"].tolist() for df in dfs]

        return queries, docs, tags

    if dataset == "stackexchange":
        if tags == "all":
            tags = [
                "history",
                "crypto",
                "chess",
                "cooking",
                "astronomy",
                "fitness",
                "anime",
                "literature",
            ]
        else:
            tags = tags.split(",")

        dfs = [load_stackexchange(tag) for tag in tags]
        queries = [df["title_body"] for df in dfs]
        docs = [df["upvoted_answer"] for df in dfs]

        return queries, docs, tags
