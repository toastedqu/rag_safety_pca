import numpy as np


def set_seed(seed):
    np.random.seed(seed)


def save_to_npy(embeddings, model, information, tags):
    for tag, d_emb in zip(tags, embeddings):
        np.save(fr"cache/{information}_{model}_{tag}.npy", d_emb)


def load_from_npy(model, information, tags):
    embeddings = []
    for tag in tags:
        embeddings.append(np.load(fr"cache/{information}_{model}_{tag}.npy"))
    return embeddings


def save_text_file(data, method, tag, model_name: str, metric: str):
    with open(f"cache/{method}_{tag}_{model_name}_{metric}.txt", "w") as f:
        for item, value in data.items():
            f.write(f"{item}\t{value}\n")
