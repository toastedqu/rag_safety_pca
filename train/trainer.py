import logging
import sys

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Trainer:
    """
    Trainer class to train and evaluate classifiers

    Attributes
    ----------
    clfs : dict
        dictionary of classifiers
    params : dict
        dictionary of classifiers parameters
    trained_clfs : dict
        dictionary of trained classifiers

    Args
    ----
    classifiers : dict
        dictionary of classifiers
    classifiers_params : dict
        dictionary of classifiers parameters
    """

    def __init__(self, classifiers: dict, classifiers_params: dict):
        self.clfs = classifiers
        self.params = classifiers_params
        self.trained_clfs = {}

    def update_param(self, clf: str, param_name: str, param_value) -> None:
        """
        Update a specific parameter of a classifier
        """
        if clf not in self.params:
            raise ValueError(f"Classifier {clf} not found")

        self.params[clf].update({param_name: param_value})

    def update_best_clf(self, classifier: str, clf: object) -> None:
        """
        Update the best classifier into the trained_clfs dictionary
        """
        self.trained_clfs.update({classifier: clf})

    def train(
        self, classifier: str, X_train: np.ndarray, y_train: np.ndarray, params: list
    ):
        if classifier == "eball":
            self.update_param(classifier, "radius", params[0])

        if classifier == "ecube":
            self.update_param(
                classifier, "sides", [params[0] for _ in range(params[1])]
            )

        if classifier == "erect":
            self.update_param(classifier, "sides", [params[0], params[1]])

        clf = self.clfs[classifier](**self.params[classifier])
        clf.fit(X_train, y_train)

        return clf

    def search_hyperparameters(
        self,
        classifier: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        pcas: np.ndarray,
        radius_search: list,
        seed=42,
    ) -> dict:
        """
        Search for the best hyperparameters for the classifier
        """
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )

        top_pcs = list(range(len(pcas)))

        results = {}
        if classifier in ("eball", "ecube"):
            for i, radius in enumerate(radius_search):
                for n_pcs in range(1, 6):
                    logger.info(
                        "Training classifier with radius %s and %s pcs"
                        % (radius, n_pcs)
                    )

                    selected_pcs = top_pcs[:n_pcs]

                    logger.info("Training started....")
                    clf = self.train(
                        classifier,
                        X_train[:, selected_pcs],
                        y_train,
                        params=[radius, n_pcs],
                    )

                    logger.info("Evaluating....")
                    results[(radius, n_pcs)] = self.evaluate(
                        X_valid[:, selected_pcs], y_valid, clf=clf
                    )
                    logger.info(f"Results: {results[(radius, n_pcs)]}")

        if classifier == "erect":
            for l, length in enumerate(radius_search):
                for w, width in enumerate(radius_search):
                    logger.info(
                        "Training classifier with length %s and width %s"
                        % (length, width)
                    )

                    selected_pcs = top_pcs[:2]

                    logger.info("Training started....")
                    clf = self.train(
                        classifier,
                        X_train[:, selected_pcs],
                        y_train,
                        params=[length, width],
                    )

                    logger.info("Evaluating....")
                    results[(length, width, 2)] = self.evaluate(
                        X_valid[:, selected_pcs], y_valid, clf=clf
                    )
                    logger.info(f"Validation Accuracy: {results[(length, width, 2)]}")

        if classifier in ["logreg", "svm", "gmm"]:
            logger.info("Training classifier....")
            n_pcs_bundle = list(range(1, 20)) + list(range(20, 180, 20))

            for n_pcs in n_pcs_bundle:
                logger.info(f"Training classifier with {n_pcs} pcs")
                selected_pcs = top_pcs[:n_pcs]

                logger.info("Training started....")
                clf = self.train(
                    classifier, X_train[:, selected_pcs], y_train, params=n_pcs
                )

                logger.info("Evaluating....")
                results[n_pcs] = self.evaluate(
                    X_valid[:, selected_pcs], y_valid, clf=clf
                )
                logger.info(f"Validation Accuracy: {results[n_pcs]}")

        return results

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, clf=None, classifier=None
    ) -> float:

        if clf is None and classifier is None:
            raise ValueError("Either clf or classifier must be provided")

        if clf is None and classifier not in self.trained_clfs:
            if classifier not in self.clfs:
                raise ValueError(f"Classifier {classifier} not found")
            else:
                logger.info(
                    f"The non trained classifier {classifier} will be evaluated"
                )
                clf = self.clfs[classifier](**self.params[classifier])

        if clf is None and classifier in self.trained_clfs:
            clf = self.trained_clfs[classifier]

        y_pred = clf.predict(X_test)
        score = accuracy_score(y_pred, y_test)

        recall = recall_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        f1score = f1_score(y_pred, y_test)

        logger.info(f"Recall: {round(recall, 3)}")
        logger.info(f"Precision: {round(precision, 3)}")
        logger.info(f"F1 Score: {round(f1score, 3)}")

        y_pred_proba = clf.predict_proba(X_test)
        queries = []


        i = 0
        with open("cache/test_queries.txt", 'r') as file:
            for line, prob in zip(file, y_pred_proba):
                if i < 103:
                    queries.append((line.strip(), prob[1]))
                i += 1

        sorted_queries = sorted(queries, key=lambda x: x[1], reverse=True)
        print("Cases that have possibility 1 to be in that class")
        for i in sorted_queries[:5]:
            print(i[0], "\t", i[1])

        print("-------------------------------------------------")

        print("Cases that have possibility 0 to be in that class")
        for i in sorted_queries[-15:]:
            print(i[0], "\t", i[1])

        # clustering methods don't have access to the class labels, so an accuracy lower than 0.5 indicates label flip.
        if score < 0.5:
            y_pred = [1 - c for c in y_pred]
            return accuracy_score(y_pred, y_test)

        return score
