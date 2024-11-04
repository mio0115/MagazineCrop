from typing import Optional

import torch
import numpy as np
import cv2


def to_onehot(tensor: torch.Tensor) -> torch.Tensor:
    onehot = torch.zeros_like(tensor)
    onehot.scatter(1, tensor.unsqueeze(1), 1)

    return onehot


def scharr_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    This function is to apply scharr edge detection to the given image.'
    This function works on grayscale image only.
    """

    # apply scharr edge detection
    gradient_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    # compute gradient magnitude
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    # convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    return gradient_magnitude


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


def resize_with_aspect_ratio(
    img: np.ndarray,
    tgt: Optional[np.ndarray] = None,
    target_size: tuple[int] = (512, 512),
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    This function is to resize images with given aspect ratio
    instead of directly resizing to target_size which cause distorted.
    """
    height, width = img.shape[:2]
    target_height, target_width = target_size

    aspect_ratio = width / height
    if height > width:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # resize the image
    resized_img = cv2.resize(img, (new_width, new_height), cv2.INTER_LINEAR)

    # compute padding to reach target size
    pad_height = target_height - new_height
    pad_width = target_width - new_width
    top, bottom = pad_height // 2, pad_height - (pad_height // 2)
    left, right = pad_width // 2, pad_width - (pad_width // 2)

    # pad the image and adjust target
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    resized_tgt = None
    if tgt is not None:
        resized_tgt = tgt.copy()
        resized_tgt[..., 0] = resized_tgt[..., 0] * new_width / width + left

    return padded_img, resized_tgt
