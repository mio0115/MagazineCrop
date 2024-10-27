import math

from torch import nn
from torch.nn import functional as F


class MeanSquaredLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, targets):
        logits[..., 0] = logits[..., 0].sigmoid()
        logits[..., 1] = logits[..., 1].sigmoid() * math.pi

        return F.mse_loss(logits, targets, reduction="sum").mean()
