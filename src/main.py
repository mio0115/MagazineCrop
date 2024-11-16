import os

import torch
import cv2
import numpy as np

from .remove_background.infer import PredictForeground
from .split_page.splitting import SplitPage
from .utils.arg_parser import get_parser
from .utils.misc import resize_with_aspect_ratio


class Combination(object):
    def __init__(self, predict_foreground, predict_split_coord):
        self._predict_fg = predict_foreground
        self._predict_sp = predict_split_coord

    @staticmethod
    def fix_mask(
        mask: np.ndarray, dividing_line: tuple[int, float], thresh: float = 0.9
    ) -> np.ndarray:
        line_x_coord, line_theta = dividing_line

        length = Combination.compute_line_length(mask, (line_x_coord, line_theta))

        # TODO: fill the mask on edges

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
        fixed_mask = Combination.fix_mask(fg_mask, (line_x_coord, line_theta))
        left_mask, right_mask = Combination.split_mask(
            fixed_mask, (line_x_coord, line_theta)
        )

        left_page, left_mask = Combination.drop_background(resized_img, left_mask)
        right_page, right_mask = Combination.drop_background(resized_img, right_mask)

        return (left_page, left_mask), (right_page, right_mask)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 1024, 1024
    predict_fg = PredictForeground(args, new_size=(new_width, new_height))
    predict_sp = SplitPage(args, new_size=(new_width, new_height))
    split_pages = Combination(predict_fg, predict_sp)

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

            cv2.namedWindow("Left Page", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Right Page", cv2.WINDOW_NORMAL)

            cv2.resizeWindow("Left Page", new_width, new_height)
            cv2.resizeWindow("Right Page", new_width, new_height)

            cv2.imshow("Left Page", left_page)
            cv2.imshow("Right Page", right_page)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
