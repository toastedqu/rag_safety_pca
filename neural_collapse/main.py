import logging
import sys

import click

from neural_collapse.clustering import Clustering
from neural_collapse.loader import dataset_loader
from utils import set_seed

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@click.group()
def cli():
    pass


@cli.command("cluster")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--data_path", help="path to data csv file", type=str, required=True)
@click.option("--dataset", help="dataset name", type=str, default="covid")
@click.option("--model", help="model name or path", type=str, default="bert-base-uncased")
@click.option("--k_min", help="minimum number of clusters", type=int, default=2)
@click.option("--k_max", help="maximum number of clusters", type=int, default=20)
@click.option("--output_path", help="path to save new dataset with labels", type=str, default="output_with_labels.csv")
def cluster(seed, data_path, dataset, model, k_min, k_max, output_path):
    """
    Generate embeddings and cluster queries to assign pseudo-labels.
    This function loads a dataset, extracts queries, clusters them using a BERT model,
    and saves the dataset with assigned labels to a specified output path.

    :param seed: The seed for reproducibility.
    :param data_path: Path to the input dataset CSV file.
    :param dataset: Name of the dataset to load (e.g., "covid").
    :param model: Pretrained BERT model name or path.
    :param k_min: The minimum number of clusters to try.
    :param k_max: The maximum number of clusters to try.
    :param output_path: Path to save the output dataset with labels.
    """

    set_seed(seed)

    logger.info(f"Loading dataset from {data_path} for {dataset}...")
    dataset = dataset_loader(data_path, dataset)
    logger.info(f"Dataset loaded with {len(dataset)} records.")

    queries = dataset["Query"].tolist()

    clustering = Clustering(logger, model, k_min=k_min, k_max=k_max)

    logger.info("Extracting pseudo-labels for queries from clustering...")
    labels = clustering.run(queries)

    dataset["Label"] = labels
    logger.info(f"Labels assigned. Saving to {output_path}...")
    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    cli()