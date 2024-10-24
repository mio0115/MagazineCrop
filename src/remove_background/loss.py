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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._all_cls == 1:
            conf_scores = logits.sigmoid()
        else:
            conf_scores = logits.softmax(dim=-1)

        # to one-hot
        if self._all_cls > 1:
            oh_target = torch.zeros_like(conf_scores)
            oh_target = oh_target.scatter(
                dim=-1, index=targets.unsqueeze(-1).long(), value=1
            )
        else:
            oh_target = targets.unsqueeze(-1).long()

        # calculate loss for each class
        loss = []
        for cls_ind in range(self._all_cls):
            if cls_ind == self._ignore_ind:
                continue

            loss.append(
                self._bin_dice_loss(conf_scores[..., cls_ind], oh_target[..., cls_ind])
            )

        loss = torch.stack(loss, dim=1).mean(-1).mean()

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
            self._ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flatten_logits = logits.flatten(1, 2)
        flatten_targets = targets.flatten(1, 2)

        if self._binary:
            ce_loss = self._ce_loss(flatten_logits.squeeze(-1), flatten_targets.float())
        else:
            ce_loss = self._ce_loss(flatten_logits, flatten_targets)
        dice_loss = self._dice_loss(flatten_logits, flatten_targets)

        return self._alpha * ce_loss + dice_loss
