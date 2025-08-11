import torch
import torch.nn as nn


def cross_entropy_loss(logits, labels):
    """
    Cross-entropy loss function for classification tasks.
    This function computes the cross-entropy loss between the predicted logits and the true labels.

    :param logits: Predicted logits from the model.
    :param labels: True labels for the classification task.
    :return: Cross-entropy loss value.
    """
    return nn.CrossEntropyLoss()(logits, labels)


def cluster_loss(features, labels, classifier_weight):
    """
    Cluster loss function for in-domain features.
    This function computes the clustering loss based on the normalized features and classifier weights.

    :param features: In-domain features extracted from the model.
    :param labels: True labels corresponding to the in-domain features.
    :param classifier_weight: Classifier weights used for computing the clustering loss.
    :return: Clustering loss value.
    """
    # LClu = - z^T wy
    normed_feat = nn.functional.normalize(features, dim=1)
    normed_w = nn.functional.normalize(classifier_weight, dim=1)
    wy = normed_w[labels]

    return -torch.mean(torch.sum(normed_feat * wy, dim=1))


def separation_loss(ood_features, classifier_weight):
    """
    Separation loss function for out-of-distribution features.
    This function computes the separation loss based on the normalized out-of-distribution features
    and classifier weights.

    :param ood_features: Out-of-distribution features extracted from the model.
    :param classifier_weight: Classifier weights used for computing the separation loss.
    :return: Separation loss value.
    """
    # LSep = mean(abs(z^T wi)) over all classes
    normed_feat = nn.functional.normalize(ood_features, dim=1)
    normed_w = nn.functional.normalize(classifier_weight, dim=1)
    cosine_sims = torch.abs(torch.matmul(normed_feat, normed_w.t()))

    return torch.mean(cosine_sims)