import os

import torch
import cv2
import numpy as np

from .remove_background.infer import PredictForeground
from .split_page.splitting import SplitPage
from .utils.arg_parser import get_parser
from .utils.misc import resize_with_aspect_ratio
from .fix_distortion.fix_distortion import FixDistortion


class Combination(object):
    def __init__(self, predict_foreground, predict_split_coord):
        self._predict_fg = predict_foreground
        self._predict_sp = predict_split_coord

    @staticmethod
    def fix_mask(mask: np.ndarray, thresh: float = 0.9) -> np.ndarray:
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

        # find the minimum area rectangle that encloses the polygon
        rect = cv2.minAreaRect(approx)
        # get the box points
        # the points are in the format of (x, y)
        # the order of the points is clockwise starting from the top-left corner
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        return mask

    @staticmethod
    def compute_line_length(mask, dividing_line):
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

    @staticmethod
    def split_mask(mask, dividing_line):
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

    @staticmethod
    def drop_background(image, mask):
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

        cropped_image = image[top:bottom, left:right, :]
        cropped_mask = mask[top:bottom, left:right]

        return cropped_image * cropped_mask[:, :, None], cropped_mask

    def __call__(self, image: np.ndarray, is_gray: bool = False):
        fg_mask = self._predict_fg(image, is_gray=is_gray)
        (line_x_coord, line_theta) = self._predict_sp(image)
        line_x_coord = int(line_x_coord / image.shape[1] * fg_mask.shape[1])

        resized_img, _ = resize_with_aspect_ratio(image, target_size=fg_mask.shape[:2])
        left_mask, right_mask = Combination.split_mask(
            fg_mask, (line_x_coord, line_theta)
        )
        fixed_left_mask = Combination.fix_mask(left_mask)
        fixed_right_mask = Combination.fix_mask(right_mask)

        cropped_left_page, cropped_left_mask = Combination.drop_background(
            resized_img, fixed_left_mask
        )
        cropped_right_page, cropped_right_mask = Combination.drop_background(
            resized_img, fixed_right_mask
        )

        left_page = {"image": cropped_left_page, "mask": cropped_left_mask}
        right_page = {"image": cropped_right_page, "mask": cropped_right_mask}

        return left_page, right_page


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 1024, 1024
    predict_fg = PredictForeground(args, new_size=(new_width, new_height))
    predict_sp = SplitPage(args, new_size=(new_width, new_height))
    split_pages = Combination(predict_fg, predict_sp)
    fix_distortion = FixDistortion(target_size=(new_width, new_height))

    dir_names = [
        "B6855",
        "B6960",
        "B6995",
        "C2789",
        "C2797",
        "C2811",
        "C2926",
        "C3909",
        "C3920",
    ]

    for dir_name in dir_names:
        path_to_dir = os.path.join(
            os.getcwd(), "data", "valid_data", "scanned", "images", dir_name
        )
        if not os.path.isdir(path_to_dir):
            continue

        for image_name in os.listdir(path_to_dir):
            if not image_name.endswith(".tif"):
                continue

            image = cv2.imread(os.path.join(path_to_dir, image_name))

            left_page, right_page = split_pages(image, is_gray=False)

            fixed_left_page = fix_distortion(left_page["image"], left_page["mask"])
            fixed_right_page = fix_distortion(right_page["image"], right_page["mask"])

            cv2.namedWindow("Left Page", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Right Page", cv2.WINDOW_NORMAL)

            cv2.resizeWindow("Left Page", new_width, new_height)
            cv2.resizeWindow("Right Page", new_width, new_height)

            resized_left_page, _ = resize_with_aspect_ratio(
                left_page["image"], target_size=(new_width, new_height)
            )
            resized_right_page, _ = resize_with_aspect_ratio(
                right_page["image"], target_size=(new_width, new_height)
            )

            cv2.imshow("Left Page", cv2.hconcat([resized_left_page, fixed_left_page]))
            cv2.imshow(
                "Right Page", cv2.hconcat([resized_right_page, fixed_right_page])
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
