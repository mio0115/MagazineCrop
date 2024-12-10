import os

import torch
import cv2
import numpy as np

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


def _find_max_component(mask: np.ndarray) -> list[list[tuple[int]]]:
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


class PredictForeground(object):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "rm_bg_entire_iter.pth",
        verbose: int = 0,
        new_size: tuple[int] = (1024, 1024),
    ):
        self._model_device = device
        self._model = torch.load(
            os.path.join(
                os.getcwd(),
                "src",
                "remove_background",
                "checkpoints",
                model_name,
            ),
            weights_only=False,
        )
        self._model.to(self._model_device)

        self._verbose = verbose
        self._new_size = new_size

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
        max_fg_mask = _find_max_component(fg_mask)

        return max_fg_mask
