import os

import torch
import cv2
import numpy as np

from .infer import PredictSplitCoord
from ..utils.misc import scharr_edge_detection
from ..utils.arg_parser import get_parser


class SplitPage(object):
    def __init__(self, args):
        self._predict_coord = PredictSplitCoord(args)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _get_hough_lines(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_image, (11, 11), 0)
        blurred_img = self._clahe.apply(blurred_img)
        canny_img = cv2.Canny(blurred_img, 50, 150)

        lines = cv2.HoughLinesP(
            canny_img,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=50,
            maxLineGap=500,
        ).squeeze(1)

        return lines

    @staticmethod
    def lines_suppress(height: int, width: int, lines: np.ndarray):
        all_lines: dict[tuple[int, int], list[float]] = {}
        for line in lines:
            x1, y1, x2, y2 = line

            if np.isclose(x1, x2):
                intersect_x = x1
                arc_angle = np.pi / 2
            else:
                slope = (y1 - y2) / (x1 - x2)
                bias = y1 - slope * x1
                if not np.isclose(slope, 0):
                    intersect_x = 1 / slope * (height / 2 - bias)

                arc_angle = np.arctan(slope)
                if arc_angle < 0:
                    arc_angle += np.pi
            angle = arc_angle * 180 / np.pi
            if angle < 80 or angle > 100:
                continue
            if (round(intersect_x), round(angle)) in all_lines.keys():
                all_lines[(round(intersect_x), round(angle))].append(angle)
            else:
                all_lines[(round(intersect_x), round(angle))] = [angle]

        for key, angles in all_lines.items():
            all_lines[key] = np.mean(angles)

        return all_lines

    @staticmethod
    def compute_line_dist(
        width: int,
        target_line: tuple[float, float],
        candidates: dict[tuple[int, int], float],
    ):
        line_dist = dict.fromkeys(candidates.keys(), 0)
        tgt_x_coord, tgt_arc_angle = target_line

        tgt_x_coord = tgt_x_coord / width
        tgt_arc_angle = tgt_arc_angle / np.pi
        for x_coord, angle in candidates.keys():
            line_dist[(x_coord, angle)] = (
                np.abs(tgt_x_coord - x_coord / width) ** 2
                + np.abs(tgt_arc_angle - angle / 180) ** 2
            ) * 0.5

        return line_dist

    @staticmethod
    def choose_best_line(
        lines, lines_dist: dict[tuple[int, int], float]
    ) -> tuple[int, float]:
        min_key, min_dist = None, 1.0

        for key, value in lines_dist.items():
            if value < min_dist:
                min_dist = value
                min_key = key
        x_coord, angle = min_key[0], lines[min_key]

        return x_coord, angle

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray]:
        line_segments = self._get_hough_lines(image)
        lines = SplitPage.lines_suppress(image.shape[0], image.shape[1], line_segments)

        pred_x, pred_theta = self._predict_coord(image)

        lines_dist = SplitPage.compute_line_dist(
            image.shape[1], (pred_x, pred_theta), lines
        )
        x_coord, angle = SplitPage.choose_best_line(lines=lines, lines_dist=lines_dist)

        return (x_coord, angle)


def to_scharr(image: np.ndarray, is_gray: bool = False):
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(image, (11, 11), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(blurred_img)

    scharr_img = scharr_edge_detection(image)

    return scharr_img


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    split_page = SplitPage(args)

    path_to_root = os.path.join(os.getcwd(), "data", "valid_data", "scanned", "images")

    for dir_name in os.listdir(path_to_root):
        path_to_dir = os.path.join(path_to_root, dir_name)
        if not os.path.isdir(path_to_dir):
            continue

        for file_name in os.listdir(path_to_dir):
            if not file_name.endswith(".tif"):
                continue

            im = cv2.imread(os.path.join(path_to_dir, file_name))
            height, width = im.shape[:2]
            (x_coord, angle) = split_page(im)

            im_cpy = im.copy()

            if np.isclose(angle, 90):
                cv2.line(im_cpy, (x_coord, 0), (x_coord, im.shape[0]), (0, 255, 0), 2)
            else:
                arc_angle = angle / 180 * np.pi
                slope = np.tan(arc_angle)

                x0 = x_coord - width // 2
                y0 = height // 2 - int(slope * (width // 2))
                x1 = x_coord + width // 2
                y1 = height // 2 + int(slope * (width // 2))

                cv2.line(im_cpy, (x0, y0), (x1, y1), (0, 255, 0), 2)

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 2048, 1024)
            cv2.imshow("image", cv2.hconcat([im, im_cpy]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
