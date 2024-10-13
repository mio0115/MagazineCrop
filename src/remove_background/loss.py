import torch
from torch import nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, *args, **kwargs):
        super(BinaryDiceLoss, self).__init__(*args, **kwargs)

        self._smooth = smooth

    def forward(self, conf_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        inter = torch.sum(conf_scores * targets, -1)
        union = conf_scores.sum(-1) + targets.sum(-1)

        dice = (2 * inter + self._smooth) / (union + self._smooth)

        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 0, smooth: float = 1, *args, **kwargs):
        super(MultiClassDiceLoss, self).__init__(*args, **kwargs)

        self._ignore_ind = ignore_index
        self._bin_dice_loss = BinaryDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        conf_scores = torch.softmax(logits, dim=-1)
        # reshape conf_scores and targets to (B, H * W, C)
        conf_scores = conf_scores.flatten(1, 2)
        targets = targets.flatten(1, 2)

        # to one-hot
        oh_target = torch.zeros_like(conf_scores)
        oh_target = oh_target.scatter(
            dim=-1, index=targets.unsqueeze(-1).long(), value=1
        )

        loss = []
        for cls_ind in range(conf_scores.shape[-1]):
            if cls_ind == self._ignore_ind:
                continue

            loss.append(
                self._bin_dice_loss(conf_scores[..., cls_ind], oh_target[..., cls_ind])
            )

        loss = torch.stack(loss, dim=1).mean(dim=-1).mean(dim=0)

        return loss
