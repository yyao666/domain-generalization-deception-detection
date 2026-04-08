import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (SimCLR)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):

        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)

        similarity_matrix = torch.matmul(
            representations,
            representations.T
        )

        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=similarity_matrix.device
        )

        similarity_matrix = similarity_matrix / self.temperature

        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)

        denominator = torch.exp(similarity_matrix).sum(dim=1)

        loss = -torch.log(
            torch.exp(positives) / denominator
        )

        return loss.mean()
