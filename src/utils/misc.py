from typing import Optional, Callable
import time

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
    Note that the order of the target_size is (width, height).
    """
    height, width = img.shape[:2]
    target_width, target_height = target_size

    if height * width <= target_height * target_width:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA

    aspect_ratio = width / height
    # case 1
    new_height1 = target_height
    new_width1 = int(target_height * aspect_ratio)
    # case 2
    new_width2 = target_width
    new_height2 = int(target_width / aspect_ratio)
    # note that either case 1 or case 2 would be valid
    # valid conditions are that new_width <= target_width and new_height <= target_height
    if new_width1 <= target_width and new_height2 <= target_height:
        if new_height1 * new_width1 < new_height2 * new_width2:
            new_width, new_height = new_width2, new_height2
        else:
            new_width, new_height = new_width1, new_height1
    elif new_width1 <= target_width:
        new_width, new_height = new_width1, new_height1
    else:
        new_width, new_height = new_width2, new_height2

    # resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation)

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


def reorder_coordinates(coords: np.ndarray):
    # reorder the coordinates based on the following order:
    # top-left, top-right, bottom-right, bottom-left
    dist = coords.sum(1)
    top_left = coords[np.argmin(dist)]
    bottom_right = coords[np.argmax(dist)]

    diff = np.diff(coords, 1)
    top_right = coords[np.argmin(diff)]
    bottom_left = coords[np.argmax(diff)]

    return np.stack([top_left, top_right, bottom_right, bottom_left])


def timeit(func: Callable):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start} seconds")
        return result

    return wrapper
