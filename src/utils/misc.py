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
    for line in lines:
        (x1, y1), (x2, y2) = line["points"]
        if x1 == x2:
            intersections.append((x1, np.pi / 2))
            continue
        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1

        slope, bias = (y1 - y2) / (x1 - x2), y1 - (y1 - y2) / (x1 - x2) * x1

        if slope > 0:
            intersections.append((1 / slope * (height / 2 - bias), np.arctan(slope)))
        else:
            intersections.append(
                (
                    1 / slope * (height / 2 - bias),
                    np.pi - np.arctan(np.abs(slope)),
                )
            )

    return intersections
