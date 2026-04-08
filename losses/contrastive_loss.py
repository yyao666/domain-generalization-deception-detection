import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Basic contrastive loss
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(
            features,
            features.T
        ) / self.temperature

        labels = labels.unsqueeze(1)

        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.ones_like(mask) - torch.eye(
            mask.shape[0],
            device=mask.device
        )

        mask = mask * logits_mask

        exp_sim = torch.exp(similarity_matrix) * logits_mask

        log_prob = similarity_matrix - torch.log(
            exp_sim.sum(dim=1, keepdim=True)
        )

        mean_log_prob_pos = (
            mask * log_prob
        ).sum(dim=1) / mask.sum(dim=1)

        loss = -mean_log_prob_pos.mean()

        return loss
