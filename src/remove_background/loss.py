from typing import Optional

import torch
import numpy as np
from torch import nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, *args, **kwargs):
        super(BinaryDiceLoss, self).__init__(*args, **kwargs)

        self._smooth = smooth

    def forward(
        self,
        conf_scores: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones_like(targets)

        inter = torch.sum(conf_scores * targets * weights, dim=(-2, -1))
        union = torch.sum(weights * conf_scores, dim=(-2, -1)) + torch.sum(
            weights * targets, dim=(-2, -1)
        )

        dice = (2 * inter + self._smooth) / (union + self._smooth)

        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        ignore_index: int = -1,
        smooth: float = 1,
        *args,
        **kwargs,
    ):
        super(MultiClassDiceLoss, self).__init__(*args, **kwargs)

        self._ignore_ind = ignore_index
        # add 1 for background class
        # note that if number_of_classes == 1, then we have binary classification
        self._all_cls = number_of_classes + (1 if number_of_classes > 1 else 0)
        self._bin_dice_loss = BinaryDiceLoss(smooth=smooth)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self._all_cls == 1:
            conf_scores = logits.sigmoid()
        else:
            conf_scores = logits.softmax(dim=1)

        # to one-hot
        if self._all_cls > 1:
            oh_target = torch.zeros_like(conf_scores)
            oh_target = oh_target.scatter(
                dim=1, index=targets.unsqueeze(-1).long(), value=1
            )
        else:
            oh_target = targets.unsqueeze(-1).long()

        # calculate loss for each class
        losses = []
        for cls_ind in range(self._all_cls):
            if cls_ind == self._ignore_ind:
                continue

            loss = self._bin_dice_loss(
                conf_scores[:, cls_ind], oh_target[:, cls_ind], weights
            )
            losses.append(loss)

        loss = torch.stack(losses).mean()

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(WeightedCrossEntropyLoss, self).__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Predicted logits, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth class indices, shape (N, H, W).
            weights (torch.Tensor): Per-pixel weights, shape (N, H, W).

        Returns:
            torch.Tensor: Weighted cross-entropy loss.
        """
        if weights is None:
            weights = torch.ones_like(targets)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(
            logits, dim=1
        )  # Shape: (N, C, H, W)

        # Gather log probabilities corresponding to the target classes
        targets = targets.unsqueeze(1).to(torch.int64)  # Shape: (N, 1, H, W)
        target_log_probs = log_probs.gather(1, targets)  # Shape: (N, 1, H, W)

        # Remove the extra dimension
        target_log_probs = target_log_probs.squeeze(1)  # Shape: (N, H, W)

        # Compute weighted loss
        loss = -weights * target_log_probs  # Shape: (N, H, W)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComboLoss(nn.Module):
    def __init__(self, number_of_classes: int, alpha: float = 0.5, *args, **kwargs):
        super(ComboLoss, self).__init__(*args, **kwargs)

        self._dice_loss = MultiClassDiceLoss(number_of_classes=number_of_classes)
        self._alpha = alpha
        self._binary = number_of_classes == 1

        if number_of_classes == 1:
            self._ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
        else:
            self._ce_loss = WeightedCrossEntropyLoss()

    def forward(
        self,
        logits: list[torch.Tensor],
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(logits, list):
            logits = [logits]

        if weights is None:
            weights = torch.ones_like(targets)

        ce_loss, dice_loss = [], []
        for logit in logits:
            flatten_logits = logit.flatten(1, 2)
            flatten_targets = targets.flatten(1, 2)

            if self._binary:
                ce_loss.append(
                    self._ce_loss(flatten_logits.squeeze(-1), flatten_targets.float())
                )
            else:
                ce_loss.append(self._ce_loss(logit.squeeze(dim=-1), targets, weights))
            dice_loss.append(self._dice_loss(logit.squeeze(dim=-1), targets, weights))

        ce_loss = torch.stack(ce_loss).mean()
        dice_loss = torch.stack(dice_loss).mean()

        return self._alpha * ce_loss + dice_loss
