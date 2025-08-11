import torch
import torch.nn as nn
from transformers import BertModel


class BertForFeatureSeparation(nn.Module):

    def __init__(self, num_classes, model, feature_dim=768):
        """
        Initializes the BertForFeatureSeparation model.

        :param num_classes: Number of classes for classification.
        :param feature_dim: Dimension of the feature space (default is 768 for BERT).
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model)
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)  # weights used in losses

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        :param input_ids: Input token IDs for the BERT model.
        :param attention_mask: Attention mask for the BERT model.
        :return: logits: Output logits from the classifier, cls_emb: CLS token embeddings.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_emb)

        return logits, cls_emb
