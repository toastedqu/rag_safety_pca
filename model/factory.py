from typing import Tuple

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model.ecn_classifer import EpsilonCubeNeighborsClassifier

clf_dict = {
    "eball": RadiusNeighborsClassifier,
    "ecube": EpsilonCubeNeighborsClassifier,
    "erect": EpsilonCubeNeighborsClassifier,
    "logreg": LogisticRegression,
    "svm": SVC,
    "gmm": GaussianMixture,
}
clf_args_dict = {
    "eball": {"outlier_label": 0},
    "ecube": {"outlier_label": 0},
    "erect": {"outlier_label": 0},
    "logreg": {},
    "svm": {"kernel": "linear"},
    "gmm": {"n_components": 2},
}


def create_classifier(methods: str) -> Tuple[dict, dict, list]:
    """
    It creates the classifiers and their parameters
    :param methods: str of methods to be used
    :return:
        classifiers: dict of classifiers
        parameters: dict of parameters
        methods: list of methods
    """
    if methods == "all":
        classifiers = clf_dict
        parameters = clf_args_dict
    else:
        classifiers = {method: clf_dict[method] for method in methods.split(",")}
        parameters = {method: clf_args_dict[method] for method in methods.split(",")}

    methods = list(classifiers.keys())

    return classifiers, parameters, methods


def create_pca(embeddings: list) -> PCA:
    """
    It creates the PCA model
    :param embeddings: list of embeddings
    :return:
        PCA model
    """
    doc_embeddings = StandardScaler().fit_transform(embeddings)
    pca = PCA()
    pca.fit(doc_embeddings)
    return pca
