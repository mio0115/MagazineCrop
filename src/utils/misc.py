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


def lines_to_coord(lines, height: int, width: int):
    intersections = []
    epsilon = 1e-6

    for line in lines:
        (x1, y1), (x2, y2) = line["points"]

        # handle vertical line case
        if abs(x1 - x2) < epsilon:
            intersections.append((x1, np.pi / 2))
            continue

        # compute slope and bias
        slope = (y1 - y2) / (x1 - x2)
        bias = y1 - slope * x1

        # compute intersection x at y = height / 2
        x_intersect = 1 / slope * (height / 2 - bias)
        x_intersect = min(max(x_intersect, 0), width)

        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle < 0:
            # ensure that angle is in [0, np.pi]
            angle += np.pi

        intersections.append((x_intersect, angle))

    return intersections
