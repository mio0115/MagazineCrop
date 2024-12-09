import os
import math

import torch
from torch import nn
import cv2
import numpy as np

from .model.model import build_model
from ..utils.arg_parser import get_parser
from ..utils.misc import resize_with_aspect_ratio

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


class PredictSplitCoord(object):
    def __init__(self, args, new_size: tuple[int] = (1024, 1024)):
        if hasattr(args, "device"):
            self._model_device = args.device
        elif hasattr(args, "use_gpu"):
            self._model_device = "cuda" if args.use_gpu else "cpu"
        else:
            print("Use either get_parser('user') or get_parser('dev')")

        if hasattr(args, "sp_pg_model_name"):
            self._model = torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "split_page",
                    "checkpoints",
                    args.sp_pg_model_name,
                ),
                weights_only=False,
            )
        else:
            self._model = torch.load(
                os.path.join(
                    os.getcwd(), "src", "split_page", "checkpoints", "sp_pg_mod.pth"
                ),
                weights_only=False,
            )
        self._model.to(self._model_device)

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
