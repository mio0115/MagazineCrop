import math

from torch import nn
from torch.nn import functional as F


class MeanSquaredLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, targets):
        logits = logits.sigmoid()

        return F.mse_loss(logits, targets, reduction="none").sum(-1).mean()
