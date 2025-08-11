from itertools import cycle
import logging
import sys

import click
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from neural_collapse.clustering import Clustering
from neural_collapse.loader import dataset_loader, load_dataloader
from neural_collapse.model import BertForFeatureSeparation
from neural_collapse.predictor import Predictor
from neural_collapse.trainer import NCTrainer
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


@cli.command("create_datasets")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--in_domain_path", help="path to in-domain data csv file", type=str, required=True)
@click.option("--out_domain_path", help="path to out-of-domain data csv file", type=str, required=True)
@click.option("--test_size", help="size of the test set", type=float, default=0.2)
def create_datasets(seed, in_domain_path, out_domain_path, test_size):
    """
    Create in-domain and out-of-domain datasets for training and testing.

    :param seed: Seed for reproducibility.
    :param in_domain_path: Path to the in-domain dataset CSV file.
    :param out_domain_path: Path to the out-of-domain dataset CSV file.
    :param test_size: Proportion of the dataset to include in the test split (default is 0.2).
    """
    set_seed(seed)

    logger.info(f"Loading in-domain dataset from {in_domain_path}...")
    id_dataset = pd.read_csv(in_domain_path).drop_duplicates()
    logger.info(f"Dataset loaded with {len(id_dataset)} records.")

    logger.info(f"Loading out-of-domain dataset from {out_domain_path}...")
    ood_dataset = pd.read_csv(out_domain_path).drop_duplicates()
    ood_dataset["Label"] = -1  # Assign a default label for OOD data
    logger.info(f"Out-of-domain dataset loaded with {len(ood_dataset)} records.")

    id_train, id_test = train_test_split(id_dataset, test_size=test_size, random_state=seed)
    ood_train, ood_test = train_test_split(ood_dataset, test_size=test_size, random_state=seed)

    logger.info(f"Train-test split done. In-domain train size: {len(id_train)}, test size: {len(id_test)}.")
    logger.info(f"Out-of-domain train size: {len(ood_train)}, test size: {len(ood_test)}.")

    logger.info("Saving datasets...")
    id_train.to_csv("id_train.csv", index=False)
    id_test.to_csv("id_test.csv", index=False)
    ood_train.to_csv("ood_train.csv", index=False)
    ood_test.to_csv("ood_test.csv", index=False)


@cli.command("train")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--data_path", help="path to data directory containing in-domain and out-of-domain datasets", type=str, required=True)
@click.option("--model", help="pretrained BERT model name or path", type=str, default="bert-base-uncased")
@click.option("--batch_size", help="batch size for training", type=int, default=32)
@click.option("--device", help="device to run the training on (e.g., 'cpu' or 'cuda')", type=str, default="cpu")
@click.option("--learning_rate", help="learning rate for the optimizer", type=float, default=2e-5)
@click.option("--alpha", help="weight for the clustering loss", type=float, default=1)
@click.option("--beta", help="weight for the separation loss", type=float, default=1)
@click.option("--lambda_oe", help="weight for the out-of-distribution entropy loss", type=float, default=0.5)
@click.option("--n_epochs", help="number of training epochs", type=int, default=1)
@click.option("--output_path", help="path to save the trained model", type=str, default="trained_model.pth")
def train(seed, data_path, model, batch_size, device, learning_rate, alpha, beta, lambda_oe, n_epochs, output_path):
    """
    Train the model using in-domain and out-of-distribution datasets.

    :param seed: Seed for reproducibility.
    :param data_path: Path to the directory containing in-domain and out-of-domain datasets.
    :param model: Pretrained BERT model name or path.
    :param batch_size: Batch size for training.
    :param device: Device to run the training on (e.g., 'cpu' or 'cuda').
    :param learning_rate: Learning rate for the optimizer.
    :param alpha: The weight for the clustering loss.
    :param beta: The weight for the separation loss.
    :param lambda_oe: The weight for the out-of-distribution entropy loss.
    :param n_epochs: Number of training epochs.
    :param output_path: Path to save the trained model.
    """
    set_seed(seed)

    logger.info(f"Loading datasets from {data_path}...")
    id_train = pd.read_csv(data_path + "/id_train.csv")
    ood_train = pd.read_csv(data_path + "/ood_train.csv")

    logger.info(f"In-domain train size: {len(id_train)}, Out-of-domain train size: {len(ood_train)}.")

    logger.info("Creating dataloaders...")
    id_dataloader = load_dataloader(id_train, model, batch_size)
    ood_dataloader = load_dataloader(ood_train, model, batch_size)

    labels = id_train["Label"].unique().tolist()
    num_classes = len(labels)
    logger.info(f"Number of classes: {num_classes}")

    logger.info("Initializing model...")
    model = BertForFeatureSeparation(num_classes, model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ood_dataloader_cyc = cycle(ood_dataloader)

    logger.info("Starting training...")
    trainer = NCTrainer(logger, model, optimizer, alpha=alpha, beta=beta, lambda_oe=lambda_oe, n_epochs=n_epochs, device=device)
    trainer.train(id_dataloader, ood_dataloader_cyc)
    logger.info("Training completed.")

    logger.info("Saving model...")
    torch.save(trainer.model.state_dict(), output_path)

    logger.info("Finding ideal threshold for OOD detection...")
    predictor = Predictor(model, device)
    threshold = predictor.find_threshold(id_dataloader, ood_dataloader)

    logger.info(f"Ideal threshold for OOD detection: {threshold}")
    logger.info("Saving threshold...")
    pd.Series([threshold]).to_csv("ideal_threshold.csv", index=False)


@cli.command("test")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--test_data_path", help="path to test dataset directory", type=str, required=True)
@click.option("--model_path", help="path to the trained model", type=str, required=True)
@click.option("--model", help="pretrained BERT model name or path", type=str, default="bert-base-uncased")
@click.option("--batch_size", help="batch size for testing", type=int, default=32)
@click.option("--device", help="device to run the testing on (e.g., 'cpu' or 'cuda')", type=str, default="cpu")
@click.option("--num_classes", help="number of classes in the dataset", type=int, default=2)
def test(seed, test_data_path, model_path, model, batch_size, device, num_classes):
    """
    Test the trained model on a specified test dataset.

    :param seed: Seed for reproducibility.
    :param test_data_path: Path to the directory containing the test dataset.
    :param model_path: Path to the trained model file.
    :param model: Pretrained BERT model name or path.
    :param batch_size: Batch size for testing.
    :param device: Device to run the testing on (e.g., 'cpu' or 'cuda').
    :param num_classes: Number of classes in the dataset.
    """
    set_seed(seed)

    logger.info(f"Loading test dataset from {test_data_path}...")

    try:
        id_test = pd.read_csv(test_data_path + "/id_test.csv")
        ood_test = pd.read_csv(test_data_path + "/ood_test.csv")

    except:
        data = pd.read_csv(test_data_path)
        id_test = data[data["Label"] == 0]
        ood_test = data[data["Label"] == 1]

    logger.info("Extracting threshold...")
    threshold = pd.read_csv("ideal_threshold.csv").iloc[0, 0]
    logger.info(f"Threshold for OOD detection: {threshold}")

    logger.info("Creating dataloaders...")
    id_dataloader = load_dataloader(id_test, model, batch_size)
    ood_dataloader = load_dataloader(ood_test, model, batch_size)

    logger.info("Loading model...")
    model = BertForFeatureSeparation(num_classes, model).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logger.info("Starting testing...")
    predictor = Predictor(model, device)

    metrics = predictor.predict(id_dataloader, ood_dataloader, threshold)

    logger.info("Testing completed.")
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    cli()