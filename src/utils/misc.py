import torch
import numpy as np
import cv2


def to_onehot(tensor: torch.Tensor) -> torch.Tensor:
    onehot = torch.zeros_like(tensor)
    onehot.scatter(1, tensor.unsqueeze(1), 1)

    return onehot


def polygon_to_mask(polygon, height: int, width: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, np.int32)
    bin_mask = cv2.fillConvexPoly(mask, pts, 1).astype(np.int64)

    return bin_mask
