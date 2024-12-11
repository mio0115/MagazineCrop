from typing import Callable, Optional

import cv2
import numpy as np

from ..utils.misc import resize_with_aspect_ratio


def _drop_background(image, masks: list[np.ndarray]):
    pages = []
    for mask in masks:
        # we need to drop the background from the image based on the mask
        for top in range(mask.shape[0]):
            if np.any(mask[top]):
                break
        for bottom in range(mask.shape[0] - 1, -1, -1):
            if np.any(mask[bottom]):
                break
        for left in range(mask.shape[1]):
            if np.any(mask[:, left]):
                break
        for right in range(mask.shape[1] - 1, -1, -1):
            if np.any(mask[:, right]):
                break

        pages.append(
            {
                "image": image[top:bottom, left:right, :],
                "mask": mask[top:bottom, left:right],
            }
        )

    return pages


def _fix_mask(masks: list[np.ndarray], thresh: float = 0.9) -> np.ndarray:
    fixed_masks = []
    for mask in masks:
        amplified_mask = (mask.copy() * 255).astype(np.uint8)

        # find the largest contour in the mask
        # note that there is only one component in the mask
        contours, _ = cv2.findContours(
            amplified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)

        # approximate the largest contour with a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        fixed_mask = cv2.fillPoly(mask.astype(np.uint8), [approx], 1)
        fixed_masks.append(fixed_mask)

    return fixed_masks


def _compute_line_length(mask, dividing_line):
    line_x_coord, line_theta = dividing_line

    true_count = np.zeros(mask.shape[1], dtype=np.int32)
    if np.isclose(line_theta, 90):
        true_count = np.sum(mask, 0)
    else:
        line_slope = np.tan(np.deg2rad(line_theta))
        for y in range(mask.shape[0]):
            # we find the delta_x for each y
            delta_x = -1 * int((y - mask.shape[0] / 2) / line_slope)
            piece_mask = mask[
                y, max(0, delta_x) : min(mask.shape[1], mask.shape[1] + delta_x)
            ]

            if delta_x < 0:
                piece_mask = np.pad(
                    piece_mask, (-delta_x, 0), "constant", constant_values=0
                )
            else:
                piece_mask = np.pad(
                    piece_mask, (0, delta_x), "constant", constant_values=0
                )
            true_count += piece_mask
    return np.median(true_count[true_count > 0])


def _split_mask(mask, dividing_line):
    line_x_coord, line_theta = dividing_line

    if np.isclose(line_theta, 90):
        # if dividing line is vertical, we can simply split the mask into two parts
        # then we pad the divided mask with zeros to make it rectangle
        left_mask = np.pad(
            mask[:, :line_x_coord],
            pad_width=((0, 0), (0, mask.shape[1] - line_x_coord)),
            mode="constant",
            constant_values=0,
        )
        right_mask = np.pad(
            mask[:, line_x_coord:],
            pad_width=((0, 0), (line_x_coord, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        left_mask, right_mask = [], []
        line_slope = np.tan(np.deg2rad(line_theta))
        # we pad the divided mask with zeros to make it rectangle
        for y in range(mask.shape[0]):
            sep_pt = line_x_coord + int((y - mask.shape[0] / 2) / line_slope)
            left_mask.append(
                np.pad(
                    mask[y, :sep_pt],
                    pad_width=(0, mask.shape[1] - sep_pt),
                    mode="constant",
                    constant_values=0,
                )
            )
            right_mask.append(
                np.pad(
                    mask[y, sep_pt:],
                    pad_width=(sep_pt, 0),
                    mode="constant",
                    constant_values=0,
                )
            )
        # finally, we return the left and right mask
        left_mask = np.stack(left_mask)
        right_mask = np.stack(right_mask)

    return left_mask, right_mask


class Combination(object):
    def __init__(
        self,
        predict_foreground,
        predict_split_coord,
        num_pages: int = 2,
        verbose: int = 0,
        save_mask_fn: Optional[Callable] = None,
    ):
        self._num_pages = num_pages
        self._verbose = verbose

        self._predict_fg = predict_foreground
        self._predict_sp = predict_split_coord
        self._save_mask_fn = save_mask_fn
        self._original_shape = None

    def mask_recovery(self, masks: list[np.ndarray], padding: tuple[int]):
        top, bottom, left, right = padding
        recovered_masks = []

        for mask in masks:
            # drop the padding
            no_pad_mask = (
                mask[
                    top : (-bottom if bottom > 0 else None),
                    left : (-right if right > 0 else None),
                ]
                * 255
            ).astype(np.uint8)
            # add additional padding to make the border away from edges
            padding_mask = np.pad(
                no_pad_mask, ((100, 100), (100, 100)), "constant", constant_values=0
            )

            # find the largest contour in the mask
            contours, _ = cv2.findContours(
                padding_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest_contour = max(contours, key=cv2.contourArea)
            # approximate the largest contour with a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)

            # convert the coordinates back to the original size
            no_pad_coords = approx - 100
            scale_x = self._original_shape[1] / no_pad_mask.shape[1]
            scale_y = self._original_shape[0] / no_pad_mask.shape[0]
            original_coords = (no_pad_coords * [scale_x, scale_y]).astype(np.int32)
            # get the mask in the original size
            new_mask = np.zeros(shape=self._original_shape[:2], dtype=np.int32)
            original_mask = cv2.fillPoly(new_mask, [original_coords], 1)

            recovered_masks.append(original_mask)

        return recovered_masks

    def _save_masks(self, image: np.ndarray, masks: list[np.ndarray], name: str):
        if self._verbose > 0:
            for i, mask in enumerate(masks):
                self._save_mask_fn(image, mask, name=f"{name}_{i}")

    def __call__(self, image: np.ndarray, is_gray: bool = False):
        self._original_shape = image.shape

        resized_img, _, padding = resize_with_aspect_ratio(
            image, target_size=self._predict_fg._new_size, return_pad=True
        )

        # get the foreground mask
        fg_mask = self._predict_fg(resized_img, is_gray=is_gray)
        self._save_masks(resized_img, [fg_mask], "foreground_mask")

        if self._num_pages == 2:
            # get the dividing line
            (line_x_coord, line_theta) = self._predict_sp(image)
            # convert the x-coordinate to the resized image
            line_x_coord = int(line_x_coord / image.shape[1] * fg_mask.shape[1])

            # split the mask into two parts based on the dividing line
            resized_left_mask, resized_right_mask = _split_mask(
                fg_mask, (line_x_coord, line_theta)
            )
            resized_masks = [resized_left_mask, resized_right_mask]
        else:
            resized_masks = [fg_mask]
        self._save_masks(resized_img, resized_masks, "splitted_mask")
        # fix the mask by either filling the holes or removing the edges
        # currently, we only fill the holes
        # TODO: find better approach to fix the mask
        fixed_masks = _fix_mask(resized_masks)
        self._save_masks(resized_img, fixed_masks, "filled_mask")

        # resize the mask back to the original size
        masks = self.mask_recovery(
            fixed_masks,
            padding=padding,
        )
        self._save_masks(image, masks, "recovered_mask")

        # drop the background from the image
        pages = _drop_background(image, masks)
        for page in pages:
            self._save_masks(page["image"], [page["mask"]], name="page_mask")

        return pages
