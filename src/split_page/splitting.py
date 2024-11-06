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
        # apply CLAHE to enhance the contrast of the image
        enhanced_img = self._clahe.apply(gray_image)
        # apply gaussian blur to reduce noise
        # TODO: compare results with/without clahe enhancement
        blurred_img = cv2.GaussianBlur(enhanced_img, (11, 11), 0)
        # apply vertical sobel filter to enhance the vertical edges
        sobel_vertical = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_vertical = np.absolute(sobel_vertical)
        sobel_vertical_8u = np.uint8(abs_sobel_vertical)
        # normalize the Sobel image
        normalized_sobel = cv2.normalize(
            sobel_vertical_8u, None, 0, 255, cv2.NORM_MINMAX
        )
        # apply binary threshold to emphasize strong edges
        _, binary_img = cv2.threshold(normalized_sobel, 50, 255, cv2.THRESH_BINARY)
        # apply canny edge detection to detect edges
        canny_img = cv2.Canny(binary_img, 50, 150)

        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("image", 2048, 1024)
        # cv2.imshow(
        #     "image",
        #     cv2.hconcat(
        #         [enhanced_img, blurred_img, sobel_vertical_8u, binary_img, canny_img]
        #     ),
        # )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        lines = cv2.HoughLinesP(
            canny_img,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=50,
            maxLineGap=500,
        )

        if lines is not None:
            lines = lines.squeeze(1)
        else:
            lines = np.array([])

        return lines

    @staticmethod
    def lines_suppress(height: int, width: int, lines: np.ndarray):
        all_line_segments: dict[tuple[int, int], list[float]] = {}
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
            all_line_segments.setdefault((round(intersect_x), round(angle)), []).append(
                [x1, y1, x2, y2]
            )

        all_lines = dict.fromkeys(all_line_segments.keys(), 0)
        for key, pt_pairs in all_line_segments.items():
            pt1, pt2 = pt_pairs[0][:2], pt_pairs[0][2:]
            for x1, y1, x2, y2 in pt_pairs[1:]:
                if y1 < pt1[1]:
                    pt1 = (x1, y1)
                if y2 > pt2[1]:
                    pt2 = (x2, y2)
            all_lines[key] = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

        return all_lines

    @staticmethod
    def compute_line_dist(
        width: int,
        target_line: tuple[float, float],
        candidates: dict[tuple[int, int], float],
    ):
        line_dist = dict.fromkeys(candidates.keys(), 0)
        tgt_x_coord, tgt_angle = target_line

        tgt_x_coord = tgt_x_coord / width
        for x_coord, angle in candidates.keys():
            line_dist[(x_coord, angle)] = (
                np.abs(tgt_x_coord - x_coord / width) ** 2
                + np.abs(tgt_angle / 180 - angle / 180) ** 2
            ) * 0.5

        return line_dist

    @staticmethod
    def choose_best_line(
        lines, lines_dist: dict[tuple[int, int], float]
    ) -> tuple[int, float]:
        thresh_keys, thresh = [], 5e-4

        for key, dist in lines_dist.items():
            if dist < thresh:
                thresh_keys.append(key)

        key = None
        if len(thresh_keys) > 0:
            key = max(thresh_keys, key=lambda x: lines[x])

        return key

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray]:
        line_segments = self._get_hough_lines(image)
        lines = SplitPage.lines_suppress(image.shape[0], image.shape[1], line_segments)

        pred_x, pred_arc_angle = self._predict_coord(image)
        pred_angle = pred_arc_angle[0] / np.pi * 180

        lines_dist = SplitPage.compute_line_dist(
            width=image.shape[1], target_line=(pred_x, pred_angle), candidates=lines
        )
        best_coord = SplitPage.choose_best_line(lines=lines, lines_dist=lines_dist)
        if best_coord is None:
            best_coord = (round(pred_x), 90)

        return best_coord, (pred_x, pred_angle)


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
            (x_coord, angle), (pred_x_coord, pred_angle) = split_page(im)

            im_cpy = im.copy()

            if np.isclose(angle, 90):
                cv2.line(im_cpy, (x_coord, 0), (x_coord, im.shape[0]), (0, 255, 0), 3)
            else:
                arc_angle = angle / 180 * np.pi
                slope = np.tan(arc_angle)

                x0 = x_coord - width // 2
                y0 = height // 2 - int(slope * (width // 2))
                x1 = x_coord + width // 2
                y1 = height // 2 + int(slope * (width // 2))

                cv2.line(im_cpy, (x0, y0), (x1, y1), (0, 255, 0), 3)

            if np.isclose(pred_angle, 90.0):
                cv2.line(
                    im_cpy,
                    (pred_x_coord, 0),
                    (pred_x_coord, im.shape[0]),
                    (0, 0, 255),
                    3,
                )
            else:
                pred_arc_angle = pred_angle / 180 * np.pi
                pred_slope = np.tan(pred_arc_angle)

                x0 = int(pred_x_coord) - width // 2
                y0 = height // 2 - int(pred_slope * (width // 2))
                x1 = int(pred_x_coord) + width // 2
                y1 = height // 2 + int(pred_slope * (width // 2))

                cv2.line(im_cpy, (x0, y0), (x1, y1), (0, 0, 255), 3)

            cv2.namedWindow(f"{dir_name}/{file_name}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{dir_name}/{file_name}", 2048, 1024)
            cv2.imshow(f"{dir_name}/{file_name}", im_cpy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
