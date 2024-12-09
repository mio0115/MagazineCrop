import os

import torch
from torch import nn
import cv2
import numpy as np

from .model.model_unet_pp import build_model
from ..utils.arg_parser import get_parser
from ..utils.misc import resize_with_aspect_ratio, timeit

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


class PredictForeground(object):
    def __init__(self, args, new_size: tuple[int] = (1024, 1024)):
        if hasattr(args, "device"):
            self._model_device = args.device
        elif hasattr(args, "use_gpu"):
            self._model_device = "cuda" if args.use_gpu else "cpu"

        if hasattr(args, "model_name"):
            self._model = torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "remove_background",
                    "checkpoints",
                    args.rm_bg_model_name,
                ),
                weights_only=False,
            )
        else:
            self._model = torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "remove_background",
                    "checkpoints",
                    "rm_bg_entire_iter.pth",
                ),
                weights_only=False,
            )
        self._model.to(self._model_device)

        self._new_size = new_size

    @staticmethod
    def find_max_component(mask: np.ndarray) -> list[list[tuple[int]]]:
        visited = np.zeros_like(mask)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def dfs_iter(start_ind):
            nonlocal mask, visited, directions
            stack = [start_ind]
            component = [start_ind]

            while stack:
                x, y = stack.pop()
                visited[x, y] = 1
                component.append((x, y))

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < mask.shape[0]
                        and 0 <= ny < mask.shape[1]
                        and mask[nx, ny]
                        and visited[nx, ny] == 0
                    ):
                        stack.append((nx, ny))

            return component

        max_component_size, max_component = 0, None
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] and visited[x, y] == 0:
                    component = dfs_iter((x, y))
                    if len(component) > max_component_size:
                        max_component_size = len(component)
                        max_component = component

        new_mask = np.zeros_like(mask)
        for x, y in max_component:
            new_mask[x, y] = 1

        return new_mask

    def __call__(self, image: np.ndarray, is_gray: bool = False) -> np.ndarray:
        if not is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        eh_image = cv2.equalizeHist(image)

        with torch.no_grad():
            in_image = (
                torch.tensor(eh_image)[None, :, :, None].permute(0, 3, 1, 2).float()
                / 255.0
            )
            in_image = in_image.to(self._model_device)
            logits = self._model(in_image)[-1]
            is_fg_prob = logits.sigmoid().squeeze().cpu().numpy()
        fg_mask = is_fg_prob >= 0.5

        # get the largest connected component
        max_fg_mask = PredictForeground.find_max_component(fg_mask)

        return max_fg_mask


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 640, 640
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
    path_to_model = os.path.join(
        os.getcwd(), "src", "remove_background", "checkpoints", args.model_name
    )
    predict_fg = PredictForeground(args, new_size=(new_width, new_height))

    model = torch.load(path_to_model, weights_only=False)
    model.to(args.device)
    model.eval()

    for dir_name in dir_names:
        path_to_dir = os.path.join(
            os.getcwd(), "data", "valid_data", "scanned", "images", dir_name
        )
        if not os.path.isdir(path_to_dir):
            continue

        for image_name in os.listdir(path_to_dir):
            if not image_name.endswith(".tif"):
                continue

            image = cv2.imread(
                os.path.join(path_to_dir, image_name), cv2.IMREAD_GRAYSCALE
            )

            fg_mask = predict_fg(image, is_gray=True)

            masked_image = resize_with_aspect_ratio(
                image, target_size=(new_width, new_height)
            )[0]
            masked_image[~fg_mask] = 0

            mask = fg_mask.astype(np.uint8) * 255

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", new_width * 3, new_height)
            cv2.imshow("image", cv2.hconcat([masked_image, mask, masked_image]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
