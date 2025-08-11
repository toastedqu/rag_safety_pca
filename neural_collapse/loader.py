import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def dataset_loader(path, dataset):
    if dataset == "covid":
        in_domain = pd.read_csv(path, encoding="unicode_escape").drop_duplicates(["user_kp", "system_response"]).reset_index(drop=True)
        in_domain = in_domain.rename(columns={"user_kp": "Query", "system_response": "Answer"})

    elif dataset == "substance_use":
        in_domain = pd.read_csv(path)
        in_domain = in_domain[in_domain["secondary intent"] != "Covid"]
        in_domain = in_domain.rename(columns={"question": "Query", "relevant answer": "Answer"})

    else:
        raise ValueError(f"Dataset {dataset} not supported. Available datasets: covid, substance_use.")

    return in_domain


class QueryDataset(Dataset):
    def __init__(self, queries, labels, model):
        """
        Custom Dataset for handling queries and labels.

        :param queries: List of query strings.
        :param labels: List of labels corresponding to the queries.
        :param tokenizer: Tokenizer to process the queries.
        """
        self.queries = queries
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(model)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Get item by index.
        :param idx: Index of the item to retrieve.
        :return: A dictionary containing the tokenized query and its label.
        """
        item = self.tokenizer(self.queries[idx], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item


def load_dataloader(dataset, model, batch_size=32):
    """
    Create a DataLoader from the dataset.

    :param dataset: DataFrame containing queries and labels.
    :param tokenizer: Tokenizer to process the queries.
    :param batch_size: Size of the batches to be loaded.
    :return: DataLoader object for the dataset.
    """
    dataset = QueryDataset(dataset["Query"].tolist(), dataset["Label"].tolist(), model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
