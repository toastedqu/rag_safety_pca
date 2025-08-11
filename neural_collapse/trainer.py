import torch

from neural_collapse.loss import cross_entropy_loss, cluster_loss, separation_loss


class NCTrainer:
    def __init__(self, logger, model, optimizer, alpha=1.0, beta=1.0, lambda_oe=0.5, n_epochs=3, device='cpu'):
        """
        Initializes the NCTrainer with model, optimizer, and loss weights.

        :param logger: Logger for logging training progress.
        :param model: The neural network model to be trained.
        :param optimizer: The optimizer for updating model parameters.
        :param alpha: Weight for the clustering loss.
        :param beta: Weight for the separation loss.
        :param lambda_oe: Weight for the out-of-distribution entropy loss.
        :param n_epochs: Number of training epochs.
        :param device: Device to run the training on (e.g., 'cpu' or 'cuda').
        """
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta
        self.lambda_oe = lambda_oe
        self.n_epochs = n_epochs
        self.device = device

    def training_step(self, batch_id, batch_ood):
        self.model.train()
        self.optimizer.zero_grad()

        # In-domain
        logits_id, features_id = self.model(batch_id["input_ids"], batch_id["attention_mask"])

        ce_loss = cross_entropy_loss(logits_id, batch_id["labels"])
        clu_loss = cluster_loss(features_id, batch_id["labels"], self.model.classifier.weight)

        # OOD
        logits_ood, features_ood = self.model(batch_ood["input_ids"], batch_ood["attention_mask"])

        probs_ood = torch.softmax(logits_ood, dim=-1)

        loe_loss = -torch.mean(torch.sum(torch.log(probs_ood + 1e-8) / probs_ood.size(1), dim=1))  # OE Loss
        sep_loss = separation_loss(features_ood, self.model.classifier.weight)

        total_loss = ce_loss + self.alpha * clu_loss + self.lambda_oe * loe_loss + self.beta * sep_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(self, train_loader_id, train_loader_ood):
        """
        Train the model using in-domain and out-of-distribution data loaders.

        :param train_loader_id: Training data loader for in-domain samples.
        :param train_loader_ood: Training data loader for out-of-distribution samples.
        """

        for epoch in range(self.n_epochs):

            self.model.train()

            for step, batch_id in enumerate(train_loader_id):

                batch_ood = next(train_loader_ood)

                batch_id = {k: v.to(self.device) for k, v in batch_id.items()}
                batch_ood = {k: v.to(self.device) for k, v in batch_ood.items()}

                loss = self.training_step(batch_id, batch_ood)

                if step % 10 == 0:
                    print(f"[Epoch {epoch}] Step {step} | Loss: {loss:.4f}")
