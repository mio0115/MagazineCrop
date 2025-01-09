from typing import Optional, Callable
import time
import math

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
    return_pad: bool = False,
    target_size: tuple[int] = (512, 512),
    interpolation: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    This function is to resize images with given aspect ratio
    instead of directly resizing to target_size which cause distorted.
    Note that the order of the target_size is (width, height).
    """
    height, width = img.shape[:2]
    target_width, target_height = target_size

    if interpolation is None:
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

    if return_pad:
        return padded_img, resized_tgt, (top, bottom, left, right)
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


def show_mask(mask, title, window_size=(600, 600)):
    amplified_mask = (mask * 255).astype(np.uint8)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, *window_size)
    cv2.imshow(title, amplified_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def combine_images(images, padding_color=(128, 128, 128), font_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    border_thickness = 10

    shape = dict.fromkeys(images.keys())
    for key in images.keys():
        shape[key] = images[key].shape

    if images["left"].shape[0] != images["right"].shape[0]:
        padding_height = abs(images["left"].shape[0] - images["right"].shape[0])
        if images["left"].shape[0] < images["right"].shape[0]:
            images["left"] = cv2.copyMakeBorder(
                images["left"],
                0,
                padding_height,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
        else:
            images["right"] = cv2.copyMakeBorder(
                images["right"],
                0,
                padding_height,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
    vert_red_line = np.zeros(
        (images["left"].shape[0], border_thickness, 3), dtype=np.uint8
    )
    vert_red_line[:, :] = (0, 0, 255)
    cv2.putText(
        images["left"],
        "LEFT",
        (50, 50),
        font,
        font_scale,
        font_color,
        font_thickness,
    )
    cv2.putText(
        images["right"],
        "RIGHT",
        (50, 50),
        font,
        font_scale,
        font_color,
        font_thickness,
    )
    bottom_image = cv2.hconcat([images["left"], vert_red_line, images["right"]])

    if images["original"].shape[1] != bottom_image.shape[1]:
        padding_width = abs(images["original"].shape[1] - bottom_image.shape[1])
        if images["original"].shape[1] < bottom_image.shape[1]:
            images["original"] = cv2.copyMakeBorder(
                images["original"],
                0,
                0,
                0,
                padding_width,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
        elif images["original"].shape[1] > bottom_image.shape[1]:
            bottom_image = cv2.copyMakeBorder(
                bottom_image,
                0,
                0,
                0,
                padding_width,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
    hori_red_line = np.zeros(
        (border_thickness, bottom_image.shape[1], 3), dtype=np.uint8
    )
    hori_red_line[:, :] = (0, 0, 255)
    cv2.putText(
        images["original"],
        "ORIGINAL",
        (50, 50),
        font,
        font_scale,
        font_color,
        font_thickness,
    )
    combined_image = cv2.vconcat([images["original"], hori_red_line, bottom_image])

    return combined_image


def combine_images_single_page(
    images, padding_color=(128, 128, 128), font_color=(0, 0, 0)
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    border_thickness = 10

    shape = dict.fromkeys(images.keys())
    for key in images.keys():
        shape[key] = images[key].shape

    cv2.putText(
        images["page"],
        "PAGE",
        (50, 50),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

    bottom_image = images["page"]

    if images["original"].shape[1] != bottom_image.shape[1]:
        padding_width = abs(images["original"].shape[1] - bottom_image.shape[1])
        if images["original"].shape[1] < bottom_image.shape[1]:
            images["original"] = cv2.copyMakeBorder(
                images["original"],
                0,
                0,
                0,
                padding_width,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
        elif images["original"].shape[1] > bottom_image.shape[1]:
            bottom_image = cv2.copyMakeBorder(
                bottom_image,
                0,
                0,
                0,
                padding_width,
                cv2.BORDER_CONSTANT,
                value=padding_color,
            )
    hori_red_line = np.zeros(
        (border_thickness, bottom_image.shape[1], 3), dtype=np.uint8
    )
    hori_red_line[:, :] = (0, 0, 255)
    cv2.putText(
        images["original"],
        "ORIGINAL",
        (50, 50),
        font,
        font_scale,
        font_color,
        font_thickness,
    )
    combined_image = cv2.vconcat([images["original"], hori_red_line, bottom_image])

    return combined_image


def compute_resized_shape(shape: tuple[int], scale: float = 1.0):
    height, width, _ = shape

    area = height * width
    resized_area = area * scale
    new_height = int(math.sqrt(resized_area * height / width))
    new_width = int(resized_area / new_height)

    return (new_height, new_width)


def compute_dist_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
