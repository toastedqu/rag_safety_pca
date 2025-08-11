import pandas as pd


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