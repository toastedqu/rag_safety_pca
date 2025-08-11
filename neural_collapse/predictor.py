import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn


class Predictor:

    def __init__(self, model, device='cpu'):
        """
        Initializes the Predictor with a model and device.

        :param model:        The neural network model to be used for prediction.
        :param device:       The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def compute_ood_score(self, logits, features, classifier_weight):
        """
        Computes the OOD score based on logits, features, and classifier weights.
        This function combines the Maximum Softmax Probability (MSP) and cosine similarity scores
        to produce a final OOD score for each sample.

        :param logits:       The output logits from the model.
        :param features:     The features extracted from the model.
        :param classifier_weight: The weights of the classifier used for computing cosine similarity.
        :return:             A tensor containing the OOD scores for each sample.
        """
        msp = torch.max(torch.softmax(logits, dim=-1), dim=1)[0]  # Maximum Softmax Probability
        cosine_scores = torch.mean(torch.abs(
            nn.functional.normalize(features, dim=1) @
            nn.functional.normalize(classifier_weight, dim=1).T), dim=1)

        return msp + cosine_scores

    def get_ood_scores(self, dataloader, label):
        """
        Computes OOD scores for all samples in the given dataloader.

        :param dataloader: DataLoader containing the samples to compute OOD scores for.
        :param label: Label to assign to all samples in this dataloader (1 for in-domain, 0 for OOD).
        :return:         A tuple containing two lists: all_scores and all_labels.
        """
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():

            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                logits, features = self.model(inputs["input_ids"], inputs["attention_mask"])

                scores = self.compute_ood_score(logits, features, self.model.classifier.weight)

                all_scores.extend(scores.cpu().numpy())
                all_labels.extend([label] * len(scores))

        return all_scores, all_labels

    def find_threshold(self, train_loader_id, train_loader_ood):
        """
        Finds the optimal threshold for OOD detection based on training data.
        This function computes OOD scores for both in-domain and out-of-distribution training data,
        and calculates the threshold that achieves a target true positive rate (TPR) of 95%.

        :param train_loader_id: Training data loader for in-domain samples.
        :param train_loader_ood: Training data loader for out-of-distribution samples.
        :return:         The optimal threshold for OOD detection.
        """
        train_id_scores, train_id_labels = self.get_ood_scores(train_loader_id, label=1)
        train_ood_scores, train_ood_labels = self.get_ood_scores(train_loader_ood, label=0)

        train_all_scores = np.concatenate([train_id_scores, train_ood_scores])
        train_all_labels = np.concatenate([train_id_labels, train_ood_labels])

        fpr, tpr, thresholds = roc_curve(train_all_labels, train_all_scores)

        target_tpr = 0.95

        idx = np.argmin(np.abs(tpr - target_tpr))
        train_threshold = thresholds[idx]

        return train_threshold

    def extract_metrics(self, y_true, y_pred):
        """
        Extracts accuracy, AUROC, and confusion matrix from true and predicted labels.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return:         A dictionary containing accuracy, AUROC, and confusion matrix.
        """
        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": acc,
            "auroc": auroc,
            "confusion_matrix": cm
        }


    def predict(self, test_loader_id, test_loader_ood, threshold):
        """
        Predicts OOD labels for the test data using the specified threshold.
        This function computes OOD scores for both in-domain and out-of-distribution test data,
        and uses the provided threshold to classify samples as in-domain or OOD.

        :param test_loader_id: Test data loader for in-domain samples.
        :param test_loader_ood: Test data loader for out-of-distribution samples.
        :param threshold: Threshold for classifying samples as in-domain or OOD.
        :return:        A dictionary containing accuracy, AUROC, and confusion matrix for the predictions.
        """

        test_id_scores, test_id_labels = self.get_ood_scores(test_loader_id, label=1)
        test_ood_scores, test_ood_labels = self.get_ood_scores(test_loader_ood, label=0)

        test_all_scores = np.concatenate([test_id_scores, test_ood_scores])
        test_all_labels = np.concatenate([test_id_labels, test_ood_labels])

        test_preds = (test_all_scores > threshold).astype(int)

        metrics = self.extract_metrics(test_all_labels, test_preds)

        return metrics