import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class Clustering:
    """
    Clustering class that uses BERT embeddings to cluster queries.
    """
    def __init__(self, logger, model, k_min=2, k_max=20):
        """
        Initialize the Clustering class.

        :param model: Pretrained BERT model name or path.
        :param k_min: Minimum number of clusters to try.
        :param k_max: Maximum number of clusters to try.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.model = BertModel.from_pretrained(model).eval()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.logger = logger

    def embed_queries(self, queries, batch_size=32):
        """
        Embed queries using the BERT model.

        :param queries: List of queries to embed.
        :param batch_size: Size of the batch for embedding.
        :return: Embeddings of the queries as a numpy array.
        """
        embeddings = []

        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)

            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state[:, 0]  # CLS token

            embeddings.append(output.cpu().numpy())

        return np.vstack(embeddings)

    def find_best_k(self, X):
        """
        Find the optimal number of clusters using the Silhouette Score.

        :param X: Embeddings of the queries.
        :return: Optimal number of clusters (k).
        """
        best_k = self.k_min
        best_score = -1

        for k in range(self.k_min, self.k_max + 1):

            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            score = silhouette_score(X, kmeans.labels_)

            self.logger.info(f"k={k}, Silhouette={score:.4f}")

            if score > best_score:
                best_k = k
                best_score = score

        return best_k

    def run(self, queries):
        """
        Run the clustering algorithm on the provided queries.

        :param queries: List of queries to cluster.
        :return: Cluster labels for each query.
        """
        query_embeddings = self.embed_queries(queries)

        best_k = self.find_best_k(query_embeddings)
        self.logger.info(f"Best number of clusters: {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=42).fit(query_embeddings)

        return kmeans.labels_
