import os

import torch
import cv2
import numpy as np

from ..utils.arg_parser import get_parser
from ..utils.misc import resize_with_aspect_ratio, scharr_edge_detection

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


def _lines_suppress(height: int, width: int, lines: np.ndarray):
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

        intersect_x = round(intersect_x)
        if intersect_x % 10 >= 5:
            intersect_x = (intersect_x // 10 + 1) * 10
        else:
            intersect_x = intersect_x // 10 * 10
        all_line_segments.setdefault((intersect_x, round(angle)), []).append(
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


def _compute_line_dist(
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


def _choose_best_line(
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


def _to_scharr(image: np.ndarray, is_gray: bool = False):
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(image, (11, 11), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(blurred_img)

    scharr_img = scharr_edge_detection(image)

    return scharr_img


class SplitPage(object):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "sp_pg_mod.pth",
        verbose: int = 0,
        new_size: tuple[int] = (1024, 1024),
    ):
        self._predict_coord = PredictSplitCoord(
            device=device, model_name=model_name, verbose=verbose, new_size=new_size
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _get_hough_lines(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply CLAHE to enhance the contrast of the image
        enhanced_img = self._clahe.apply(gray_image)
        # apply GaussianBlur to reduce noise
        blurred_img = cv2.GaussianBlur(enhanced_img, (11, 11), 0)
        # apply vertical sobel filter to enhance the vertical edges
        sobel_vertical = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_vertical = np.absolute(sobel_vertical)
        sobel_vertical_8u = np.uint8(abs_sobel_vertical)
        # apply morphological opening to remove noise
        morph_img = cv2.morphologyEx(
            sobel_vertical_8u, cv2.MORPH_OPEN, self._morph_kernel
        )

        # apply canny edge detection to detect edges
        canny_img = cv2.Canny(morph_img, 30, 100)

        lines = cv2.HoughLinesP(
            canny_img,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=50,
            maxLineGap=5,
        )

        if lines is not None:
            lines = lines.squeeze(1)
        else:
            lines = np.array([])

        return lines

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray]:
        line_segments = self._get_hough_lines(image)
        lines = _lines_suppress(image.shape[0], image.shape[1], line_segments)

        pred_x, pred_arc_angle = self._predict_coord(image)
        pred_angle = pred_arc_angle[0] / np.pi * 180

        lines_dist = _compute_line_dist(
            width=image.shape[1], target_line=(pred_x, pred_angle), candidates=lines
        )
        best_coord = _choose_best_line(lines=lines, lines_dist=lines_dist)
        if best_coord is None:
            best_coord = (round(pred_x), 90)

        return best_coord


class PredictSplitCoord(object):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "sp_pg_mod.pth",
        verbose: int = 0,
        new_size: tuple[int] = (1024, 1024),
    ):
        self._model_device = device
        self._model = torch.load(
            os.path.join(
                os.getcwd(),
                "src",
                "split_page",
                "checkpoints",
                model_name,
            ),
            weights_only=False,
        )
        self._model.to(self._model_device)

        self._verbose = verbose
        self._new_size = new_size

    def __call__(self, image: np.ndarray) -> tuple[int]:
        height, width = image.shape[:2]

        resized_image, _ = resize_with_aspect_ratio(image, target_size=self._new_size)

        with torch.no_grad():
            in_image = (
                torch.tensor(resized_image).unsqueeze(0).permute(0, 3, 1, 2).float()
                / 255.0
            )
            in_image = in_image.to(self._model_device)
            logits = self._model(in_image)
            line_coords = logits.sigmoid().cpu().numpy()

        line_x_coord = line_coords[..., 0].item() * width
        line_theta = line_coords[..., 1] * np.pi

        return line_x_coord, line_theta


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 1024, 1024

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
    path_to_models = os.path.join(os.getcwd(), "src", "split_page", "checkpoints")

    model = torch.load(
        os.path.join(path_to_models, args.model_name), weights_only=False
    )

    model.to(args.device)
    model.eval()

    for dir_name in dir_names:
        path_to_dir = os.path.join(
            os.getcwd(), "data", "valid_data", "scanned", "images", dir_name
        )

        for file_name in os.listdir(path_to_dir):
            path_to_image = os.path.join(path_to_dir, file_name)

            cv2.namedWindow("splitted image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("splitted image", width=new_width, height=new_height)

            image = cv2.imread(os.path.join(path_to_image))

            height, width, *_ = image.shape

            resized_image, _ = resize_with_aspect_ratio(
                image, target_size=(new_width, new_height)
            )

            with torch.no_grad():
                in_image = (
                    torch.tensor(resized_image).unsqueeze(0).permute(0, 3, 1, 2).float()
                    / 255.0
                )
                in_image = in_image.to(args.device)
                logits = model(in_image)
                line_coords = logits.sigmoid().cpu().numpy()

                line_x_coord = int(line_coords[..., 0].item() * width)
                line_theta = line_coords[..., 1] * np.pi

                point_0 = (line_x_coord, height // 2)

                line_slope = np.tan(line_theta).item()

                point_1 = (line_x_coord - 1000, int(height // 2 - line_slope * 1000))
                point_2 = (line_x_coord + 1000, int(height // 2 + line_slope * 1000))
                # point_1 = (line_x_coord, int(height // 2 - 5000))
                # point_2 = (line_x_coord, int(height // 2 + 5000))

                cv2.line(
                    image, pt1=point_1, pt2=point_2, color=(0, 0, 255), thickness=3
                )

                # cv2.imshow("masked image", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
                cv2.imshow("splitted image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
