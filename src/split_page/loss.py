import torch
from torch import nn
from torch.nn import functional as F


class MeanSquaredLoss(nn.Module):
    def __init__(self, weights: list[float] = [2.0, 1.0], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        logits = logits.sigmoid()
        loss = F.mse_loss(logits, targets, reduction="none") * self._weights

        return loss.sum(-1).mean()
