from typing import Tuple

import numpy as np
import scipy
from scipy.spatial.distance import euclidean
from sklearn.utils import shuffle


def significance_test(arr1, arr2):

    return [scipy.stats.ttest_ind(arr1[:, i], arr2[:, i]).pvalue for i in range(200)]#len(arr1[0]))]

def create_features(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """
    It creates the features for the given metric
    :param a: The query embeddings
    :param b: PCA transformed document embeddings
    :param metric: metric to be used
    :return:
        Data matrix for the given metric
    """
    if metric == "proj":
        return np.dot(a, b.T)

    if metric == "dist":
        X = np.zeros((len(a), len(b)))

        for i in range(len(a)):
            for j in range(len(b)):
                X[i][j] = euclidean(a[i], b[j])

        return X


def create_sets(
    positive: np.ndarray, negative: np.ndarray, criterion="EVR", train_test_split=0.2, seed=42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    It creates the train and test sets
    :param positive: data matrix for positive class
    :param negative: data matrix for negative class
    :param train_test_split: ratio of train and test split
    :param seed: random seed
    :return:
        X_train: train data matrix
        X_test: test data matrix
        y_train: train labels
        y_test: test labels
    """
    positive = shuffle(positive.squeeze())[: len(negative)]
    negative = shuffle(negative.squeeze())[: len(positive)]

    positive_train = positive[: int(len(positive) * (1 - train_test_split))]
    positive_test = positive[int(len(positive) * (1 - train_test_split)) :]

    negative_train = negative[: int(len(negative) * (1 - train_test_split))]
    negative_test = negative[int(len(negative) * (1 - train_test_split)) :]

    if criterion == "p_values":
        p_values_train = significance_test(positive_train[:len(negative_train)], negative_train[:len(positive_train)])

        indices = np.argsort(p_values_train)

        positive_train = positive_train[:, indices]
        negative_train = negative_train[:, indices]

        positive_test = positive_test[:, indices]
        negative_test = negative_test[:, indices]

    y_train = [1] * len(positive_train) + [0] * len(negative_train)
    y_test = [1] * len(positive_test) + [0] * len(negative_test)

    X_train = np.concatenate((positive_train, negative_train))
    X_test = np.concatenate((positive_test, negative_test))

    X_train, y_train = shuffle(X_train, np.array(y_train), random_state=seed)
    X_test, y_test = shuffle(X_test, np.array(y_test), random_state=seed)

    return X_train, X_test, y_train, y_test, indices

