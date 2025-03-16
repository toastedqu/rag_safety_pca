import logging
import sys

import click
import numpy as np
import pandas as pd
from datasets.utils.logging import disable_progress_bar

from dataset.loader import load_queries_docs
from features.factory import create_features, create_sets, create_kb
from model.factory import create_classifier
from model.loader import ModelLoader
from model.pca import PCAClassifier
from train.trainer import Trainer
from train.utils import find_best_hyperparameters
from utils import (
    set_seed,
    save_to_npy,
    load_from_npy,
    save_text_file,
    plot_pcas,
    plot_accuracies,
    save_to_txt,
    load_text_file,
    plot_histograms,
)

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
            for local dataset choose tags between (covid, drugs, 4chan)
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
@click.option("--criterion", help="criterion to use", type=str, default="p_values")
def generate_datasets(
    seed, model_name, tags, metric, negative_tags, test_size, criterion
):
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

                logger.info(
                    f"Creating negative features for {tag} with tag {negative_tag}...."
                )
                negative_temp_data = create_features(
                    negative_query_embs[0], pca, metric
                )

                negative_data.append(negative_temp_data)

        logger.info(f"Creating data sets for {tag}....")
        X_train, X_test, y_train, y_test, inds = create_sets(
            data, np.concatenate(negative_data), criterion, test_size, seed
        )
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        save_to_txt(inds, tag, f"{model_name}")

    logger.info(f"Saving datasets to npy files ....")
    save_to_npy(X_trains, model_name, f"X_train_{metric}", tags)
    save_to_npy(X_tests, model_name, f"X_test_{metric}", tags)
    save_to_npy(y_trains, model_name, f"y_train_{metric}", tags)
    save_to_npy(y_tests, model_name, f"y_test_{metric}", tags)


@cli.command("build_kb")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option(
    "--negative_tags",
    help="negative tags to create sets(comma separated)",
    type=str,
    default="all",
)
@click.option(
    "--dataset", help="dataset path for positive samples", type=str, default=None
)
@click.option(
    "--negative_dataset",
    help="dataset path for negative samples",
    type=str,
    default=None,
)
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option("--test_size", help="test size", type=float, default=0.2)
@click.option("--criterion", help="criterion to use", type=str, default="p_values")
def build_kb(
    seed,
    model_name,
    tags,
    metric,
    negative_tags,
    dataset,
    negative_dataset,
    test_size,
    criterion,
):
    """
    Generate a knowledge base for training
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param tags: str dataset tags (comma separated) to generate datasets for
    :param metric: str metric to use. choose from (proj, dist)
    :param negative_tags: str negative tags to create sets (comma separated).
                            They will be used for the negative samples.
    :param dataset: str dataset path for positive samples
    :param negative_dataset: str dataset path for negative samples
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

    if dataset:
        logger.info(f"Loading embeddings from file path....")
        queries = pd.read_csv(dataset)["Query"].tolist()
        model = ModelLoader(model_name)
        query_embs = [model.encode(queries)]
    else:
        logger.info(f"Loading embeddings from npy files....")
        query_embs = load_from_npy(model_name, "queries", tags)

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    logger.info(f"Creating positive features....")
    data = create_features(query_embs[0], pcas[0], metric)

    negative_data = []

    if negative_dataset:
        logger.info(f"Loading embeddings from file path....")
        negative_queries = pd.read_csv(negative_dataset)["Query"].tolist()
        model = ModelLoader(model_name)
        negative_query_embs = [model.encode(negative_queries)]
        negative_temp_data = create_features(negative_query_embs[0], pcas[0], metric)
        negative_data.append(negative_temp_data)
    else:
        for negative_tag in negative_tags:
            if negative_tag not in tags:
                negative_query_embs = load_from_npy(
                    model_name, "queries", [negative_tag]
                )
                logger.info(f"Creating negative features with tag {negative_tag}....")
                negative_temp_data = create_features(
                    negative_query_embs[0], pcas[0], metric
                )

                negative_data.append(negative_temp_data)

    X, y, inds = create_kb(
        data, np.concatenate(negative_data), criterion, test_size, seed
    )
    save_to_txt(inds, f"kb_{tags}", f"{model_name}")

    logger.info(f"Saving datasets to npy files ....")
    save_to_npy([X], model_name, f"X_train_{metric}", [f"kb_{tags}"])
    save_to_npy([y], model_name, f"y_train_{metric}", [f"kb_{tags}"])


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
                method,
                X_tr[:, [x for x in range(best_params[-1])]],
                y_tr,
                best_params[:-1] if len(best_params) > 2 else best_params,
            )

            logger.info("Evaluating best classifier....")
            score = trainer.evaluate(
                X_te[:, [x for x in range(best_params[-1])]], y_te, clf=clf
            )

            logger.info(
                f"Test Accuracy for tag {tag} and method {method} is {round(score, 3)}"
            )


@cli.command("train_and_evaluate")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option(
    "--methods", help="method to use (comma separated)", type=str, default="all"
)
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option(
    "--test_tags",
    help="dataset tags as test set (comma separated)",
    type=str,
    default="none",
)
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option(
    "--radius", help="radius to search for eball and ecube", type=float, default=0.01
)
@click.option("--length", help="length for erect", type=str, default="0.01")
@click.option("--width", help="width for erect", type=float, default=0.01)
@click.option("--n_pcas", help="number of principal components", type=int, default=5)
@click.option(
    "--best", help="if evaluation on best hyperparameters", type=bool, default=False
)
@click.option(
    "--ind_set",
    help="indices path file for restructuring indices",
    type=str,
    default="",
)
def train_and_evaluate(
    seed,
    model_name,
    methods,
    tags,
    test_tags,
    metric,
    radius,
    length,
    width,
    n_pcas,
    best,
    ind_set,
):
    """
    Train and evaluate classifiers per use case, without hyperparameter tuning
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param methods: str methods to use (comma separated)
    :param tags: str dataset tags (comma separated) to train and evaluate classifiers for
    :param test_tags: str dataset tags as test set (comma separated) to test classifiers for
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

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    if test_tags == "all":
        test_tags = [
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
        test_tags = test_tags.split(",")

    for tag, X_tr, y_tr, pca in zip(tags, X_train, y_train, pcas):

        if "none" in test_tags:
            X_test = load_from_npy(model_name, f"X_test_{metric}", tags)[0]
            y_test = load_from_npy(model_name, f"y_test_{metric}", tags)[0]

        else:
            test_data = []
            for test_tag in test_tags:
                test_query_embs = load_from_npy(model_name, "queries", [test_tag])

                logger.info(f"Creating test features with tag {test_tag}....")
                test_temp_data = create_features(test_query_embs[0], pca, metric)

                test_data.append(test_temp_data)
            X_test = np.concatenate(test_data).squeeze()
            y_test = [0] * len(X_test)

        trainer = Trainer(classifiers, parameters)

        for method in methods:
            logger.info(f"Training classifier for tag {tag} and method {method}....")

            if method == "erect":
                params = [float(x) for x in length.split(",")]
            else:
                params = [radius, n_pcas]

            if best:
                results = load_text_file(method, tag, model_name, metric)
                best_params = eval(find_best_hyperparameters(results)[0])
                logger.info(
                    f"Best hyperparameters for tag {tag} and method {method} are {best_params}"
                )

                if not isinstance(best_params, list):
                    best_params = [best_params]

                n_pcas = best_params[-1]
                params = best_params[:-1] if len(best_params) > 2 else best_params

            clf = trainer.train(
                method, X_tr[:, [x for x in range(n_pcas)]], y_tr, params
            )

            if ind_set:
                inds = []
                with open(ind_set, "r") as f:
                    for line in f:
                        inds.append(int(line.strip()))

                X_test = X_test[:, inds]

            logger.info(f"Evaluating classifier for tag {tag} and method {method}....")
            score = trainer.evaluate(
                X_test[:, [x for x in range(n_pcas)]], y_test, clf=clf
            )

            logger.info(
                f"Test Accuracy for tag {tag} and method {method} is {round(score,3)}"
            )


@cli.command("generate_kb_distances")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--tags", help="dataset tags (comma separated)", type=str, default="all")
@click.option(
    "--test_tags",
    help="dataset tags as test set (comma separated)",
    type=str,
    default="none",
)
@click.option(
    "--test_dataset", help="dataset path for test samples", type=str, default=None
)
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option("--n_pcas", help="number of principal components", type=int, default=5)
@click.option(
    "--ind_set",
    help="indices path file for restructuring indices",
    type=str,
    default="",
)
def generate_kb_distances(
    seed, model_name, tags, test_tags, test_dataset, metric, n_pcas, ind_set
):

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

    logger.info(f"Loading datasets from npy files....")
    X_train = load_from_npy(model_name, f"X_train_{metric}", [f"kb_{tags}"])
    y_train = load_from_npy(model_name, f"y_train_{metric}", [f"kb_{tags}"])

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", tags)

    if test_tags == "all":
        test_tags = [
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

    elif test_tags == "none":
        test_tags = []

    else:
        test_tags = test_tags.split(",")

    test_data = []

    if test_dataset:
        logger.info(f"Loading embeddings from file path....")
        queries = pd.read_csv(test_dataset, header=None)[0].tolist()
        model = ModelLoader(model_name)
        query_embs = [model.encode(queries)]
        test_temp_data = create_features(query_embs[0], pcas[0], metric)
        test_data.append(test_temp_data)

    for test_tag in test_tags:
        test_query_embs = load_from_npy(model_name, "queries", [test_tag])

        logger.info(f"Creating test features with tag {test_tag}....")
        test_temp_data = create_features(test_query_embs[0], pcas[0], metric)

        test_data.append(test_temp_data)

    X_test = np.concatenate(test_data).squeeze()

    if ind_set:
        inds = []
        with open(ind_set, "r") as f:
            for line in f:
                inds.append(int(line.strip()))

        X_test = X_test[:, inds]

    classifiers, parameters, methods = create_classifier("eball")

    trainer = Trainer(classifiers, parameters)
    params = [100000000, n_pcas]

    clf = trainer.train(
        methods[0], X_train[0][:, [x for x in range(n_pcas)]], y_train[0], params
    )

    plot_histograms(clf, X_test[:, [x for x in range(n_pcas)]])


@cli.command("generate_pca_plot")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option(
    "--main_dataset", help="main dataset for pca plot", type=str, default="covid"
)
@click.option(
    "--rest_datasets",
    help="rest datasets to plot",
    type=str,
    default="all",
)
@click.option("--metric", help="metric to use", type=str, default="proj")
def generate_pca_plot(seed, model_name, main_dataset, metric, rest_datasets):
    """
    Generate datasets for training and testing
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param main_dataset: str main_dataset to generate pca plot
    :param metric: str metric to use. choose from (proj, dist)
    :param rest_datasets: str rest_datasets to create pca plots (comma separated).

    It will generate pca plots
    """
    set_seed(seed)

    if rest_datasets == "all":
        rest_datasets = [
            "covid",
            "drugs",
            "4chan",
            "llmattack",
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
        ].remove(main_dataset)
    else:
        rest_datasets = rest_datasets.split(",")

    logger.info(f"Loading embeddings from npy files....")
    query_embs = load_from_npy(model_name, "queries", [main_dataset])

    logger.info(f"Loading PCA components from npy files....")
    pcas = load_from_npy(model_name, "pca", [main_dataset])

    logger.info(f"Creating positive features for {main_dataset}....")
    data = create_features(query_embs[0], pcas[0], metric)

    total_data = []
    total_data.append(data)

    for tag in rest_datasets:
        logger.info(f"Loading embeddings from npy files....")
        query_embs = load_from_npy(model_name, "queries", [tag])

        logger.info(f"Creating positive features for {tag}....")
        data = create_features(query_embs[0], pcas[0], metric)[:515]
        total_data.append(data)

    plot_pcas(total_data, [main_dataset] + rest_datasets)


@cli.command("generate_acc_plot")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--model_name", help="model name", type=str, default="all-mpnet-base-v2")
@click.option("--dataset_to_plot", help="dataset to plot", type=str, default="covid")
@click.option(
    "--average",
    help="if average of datasets to be plotted",
    type=bool,
    default=False,
)
@click.option("--metric", help="metric to use", type=str, default="proj")
@click.option(
    "--radius_eball", help="radius to search for eball", type=str, default=0.01
)
@click.option(
    "--radius_ecube", help="radius to search for ecube", type=str, default=0.01
)
def generate_acc_plot(
    seed, model_name, dataset_to_plot, average, metric, radius_eball, radius_ecube
):
    """
    Generate datasets for training and testing
    :param seed: int seed for reproducibility
    :param model_name: str model name
    :param dataset_to_plot: str dataset (comma separated) to generate plots
    :param metric: str metric to use. choose from (proj, dist)
    :param average: bool if average is plotted
    :param radius_eball: float radius to search for eball
    :param radius_ecube: float radius to search for ecube

    It will generate accuracy plots vs #pcas
    """
    set_seed(seed)

    if dataset_to_plot == "all":
        dataset_to_plot = [
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
        dataset_to_plot = dataset_to_plot.split(",")

    plot_accuracies(
        dataset_to_plot, average, model_name, metric, radius_eball, radius_ecube
    )


if __name__ == "__main__":
    cli()
