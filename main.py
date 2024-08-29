import logging
import sys

import click
import numpy as np
from datasets.utils.logging import disable_progress_bar

from dataset.loader import load_queries_docs
from features.factory import create_features, create_sets
from model.factory import create_classifier
from model.loader import ModelLoader
from model.pca import PCAClassifier
from train.trainer import Trainer
from train.utils import find_best_hyperparameters
from utils import set_seed, save_to_npy, load_from_npy, save_text_file

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
disable_progress_bar()


@click.group()
def cli():
    pass


@cli.command("generate_embeddings")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--dataset", help="dataset name", type=str, default="stackexchange")
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
def generate_embeddings(seed, model_name, dataset, tags):
    """
    Generate embeddings for queries and documents of a specific dataset and tags
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param dataset: str dataset name. choose from stackexchange, local, msmarco
    :param tags: str dataset tags (comma separated).
            for local dataset both datasets (covid, drugs) are loaded
            for stackexchange dataset choose tags between ("history", "crypto", "chess", "cooking", "astronomy",
            "fitness", "anime", "literature")
            for msmarco dataset choose tags between ("biomedical", "music", "film", "finance", "law", "computing")

    It will generate embeddings that will be stored in cache folder
    """
    set_seed(seed)

    logger.info(
        f"Loading queries and documents from {dataset} dataset with tags {tags}"
    )
    queries, docs, tags = load_queries_docs(dataset, tags)
    logger.info("Done loading queries and documents")

    model = ModelLoader(model_name)

    logger.info(f"Generating embeddings for queries....")
    queries_embs = [model.encode(query) for query in queries]

    logger.info(f"Generating embeddings for documents....")
    docs_embs = [model.encode(doc) for doc in docs]

    logger.info(f"Saving embeddings to npy files ....")
    save_to_npy(queries_embs, model_name, "queries", tags)
    save_to_npy(docs_embs, model_name, "docs", tags)


@cli.command("generate_pcas")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
def generate_pcas(seed, model_name, tags):
    """
    Generate PCA components for embeddings of documents
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param tags: str dataset tags (comma separated) to generate PCA components for

    It will generate PCA components that will be stored in cache folder
    """
    set_seed(seed)

    if tags == "all":
        tags = [
            "covid",
            "drugs",
            "biomedical",
            "music",
            "film",
            "finance",
            "law",
            "computing",
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

    logger.info(f"Loading embeddings from npy files....")
    embeddings = load_from_npy(model_name, "docs", tags)

    pca_components = []

    for tag, emb in zip(tags, embeddings):
        logger.info(f"Generating PCA for {tag}....")
        pca = PCAClassifier()
        pca_emb = pca.fit(emb)

        pca_components.append(pca_emb.components_)

    logger.info(f"Saving PCA components to npy files ....")
    save_to_npy(pca_components, model_name, "pca", tags)


@cli.command("generate_datasets")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option(
    "--negative_tags",
    help="negative tags to create sets(comma separated)",
    type=str,
    default="all",
)
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option("--test_size", help="test size", type=float, default=0.2)
def generate_datasets(seed, model_name, tags, metric, negative_tags, test_size):
    """
    Generate datasets for training and testing
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param tags: str dataset tags (comma separated) to generate datasets for
    :param metric: str metric to use. choose from (proj, dist)
    :param negative_tags: str negative tags to create sets (comma separated).
                            They will be used for the negative samples.
    :param test_size: float test size

    It will generate datasets that will be stored in cache folder
    """
    set_seed(seed)

    if tags == "all":
        tags = [
            "covid",
            "drugs",
            "biomedical",
            "music",
            "film",
            "finance",
            "law",
            "computing",
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

    if negative_tags == "all":
        negative_tags = [
            "covid",
            "drugs",
            "biomedical",
            "music",
            "film",
            "finance",
            "law",
            "computing",
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
        negative_tags = negative_tags.split(",")

    logger.info(f"Loading embeddings from npy files....")
    query_embs = load_from_npy(model_name, "queries", tags)

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []

    for tag, query_emb, pca in zip(tags, query_embs, pcas):
        logger.info(f"Creating positive features for {tag}....")
        data = create_features(query_emb, pca, metric)

        negative_data = []
        for negative_tag in negative_tags:
            if negative_tag != tag:
                negative_query_embs = load_from_npy(
                    model_name, "queries", [negative_tag]
                )
                negative_pcas = load_from_npy(model_name, "pca", [negative_tag])

                logger.info(
                    f"Creating negative features for {tag} with tag {negative_tag}...."
                )
                negative_temp_data = create_features(
                    negative_query_embs[0], negative_pcas[0], metric
                )

                negative_data.append(negative_temp_data)

        logger.info(f"Creating data sets for {tag}....")
        X_train, X_test, y_train, y_test = create_sets(
            data, np.concatenate(negative_data), test_size, seed
        )
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    logger.info(f"Saving datasets to npy files ....")
    save_to_npy(X_trains, model_name, f"X_train_{metric}", tags)
    save_to_npy(X_tests, model_name, f"X_test_{metric}", tags)
    save_to_npy(y_trains, model_name, f"y_train_{metric}", tags)
    save_to_npy(y_tests, model_name, f"y_test_{metric}", tags)


@cli.command("search_hyperparameters")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option(
    "--radiuses",
    help="radius to search for eball and ecube",
    type=list,
    default=[0.01] + list(map(lambda x: x / 1000, range(0, 301, 20)))[1:],
)
@click.option(
    "--methods", help="method to use (comma separated)", type=str, default="all"
)
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option("--metric", help="metric to use", type=str, default="proj")
def search_hyperparameters(seed, model_name, radiuses, methods, tags, metric):
    """
    Search hyperparameters for classifiers
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param radiuses: list radius to search for the best hyperparameter
    :param methods: str method to use (comma separated). choose from (eball, ecube, erect, logreg, svm, gmm).
                        You may choose more than one.
    :param tags: str dataset tags (comma separated) to search hyperparameters for
    :param metric: str metric to use

    It will search hyperparameters for classifiers and save the results in cache folder
    """

    set_seed(seed)
    classifiers, parameters, methods = create_classifier(methods)

    if tags == "all":
        tags = [
            "covid",
            "drugs",
            "biomedical",
            "music",
            "film",
            "finance",
            "law",
            "computing",
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

    logger.info(f"Loading datasets from npy files....")
    X_train = load_from_npy(model_name, f"X_train_{metric}", tags)
    y_train = load_from_npy(model_name, f"y_train_{metric}", tags)

    X_test = load_from_npy(model_name, f"X_test_{metric}", tags)
    y_test = load_from_npy(model_name, f"y_test_{metric}", tags)

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    for tag, X_tr, y_tr, X_te, y_te, pca in zip(
        tags, X_train, y_train, X_test, y_test, pcas
    ):
        trainer = Trainer(classifiers, parameters)

        for method in methods:
            logger.info(
                f"Searching hyperparameters for tag {tag} and method {method}...."
            )
            results = trainer.search_hyperparameters(method, X_tr, y_tr, pca, radiuses)

            logger.info(f"Saving results to text file....")
            save_text_file(results, method, tag, model_name, metric)

            best_params = find_best_hyperparameters(results)
            logger.info(
                f"Best hyperparameters for tag {tag} and method {method} are {best_params}"
            )

            logger.info("Training best classifier....")
            clf = trainer.train(
                method, X_tr[:, [x for x in range(best_params[-1])]], y_tr, best_params
            )

            logger.info("Evaluating best classifier....")
            score = trainer.evaluate(
                X_te[:, [x for x in range(best_params[-1])]], y_te, clf=clf
            )

            logger.info(f"Test Accuracy for tag {tag} and method {method} is {score}")


@cli.command("train_and_evaluate")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option(
    "--methods", help="method to use (comma separated)", type=str, default="all"
)
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option(
    "--radius", help="radius to search for eball and ecube", type=float, default=0.01
)
@click.option("--length", help="length for erect", type=float, default=0.01)
@click.option("--width", help="width for erect", type=float, default=0.01)
@click.option("--n_pcas", help="number of principal components", type=int, default=5)
def train_and_evaluate(
    seed, model_name, methods, tags, metric, radius, length, width, n_pcas
):
    """
    Train and evaluate classifiers per use case, without hyperparameter tuning
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param methods: str methods to use (comma separated)
    :param tags: str dataset tags (comma separated) to train and evaluate classifiers for
    :param metric: str metric to use
    :param radius: float radius to search for eball and ecube
    :param length: float length for erect
    :param width: float width for erect
    :param n_pcas: int number of principal components

    It will train and evaluate classifiers for the given tags, methods and input hyperparameters and will output the result
    """

    set_seed(seed)
    classifiers, parameters, methods = create_classifier(methods)

    if tags == "all":
        tags = [
            "covid",
            "drugs",
            "biomedical",
            "music",
            "film",
            "finance",
            "law",
            "computing",
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

    logger.info(f"Loading datasets from npy files....")
    X_train = load_from_npy(model_name, f"X_train_{metric}", tags)
    y_train = load_from_npy(model_name, f"y_train_{metric}", tags)

    X_test = load_from_npy(model_name, f"X_test_{metric}", tags)
    y_test = load_from_npy(model_name, f"y_test_{metric}", tags)

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    for tag, X_tr, y_tr, X_te, y_te, pca in zip(
        tags, X_train, y_train, X_test, y_test, pcas
    ):
        trainer = Trainer(classifiers, parameters)

        for method in methods:
            logger.info(f"Training classifier for tag {tag} and method {method}....")

            if method == "erect":
                params = [length, width]
                n_pcas = 2
            else:
                params = [radius, n_pcas]

            clf = trainer.train(
                method, X_tr[:, [x for x in range(n_pcas)]], y_tr, params
            )

            logger.info(f"Evaluating classifier for tag {tag} and method {method}....")
            score = trainer.evaluate(X_te[:, [x for x in range(n_pcas)]], y_te, clf=clf)

            logger.info(f"Test Accuracy for tag {tag} and method {method} is {score}")


if __name__ == "__main__":
    cli()
