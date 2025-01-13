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

    def __call__(
        self, image: np.ndarray, is_gray: bool = False, return_prob: bool = False
    ) -> np.ndarray:
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
        if return_prob:
            return is_fg_prob
        fg_mask = is_fg_prob >= 0.5

        # get the largest connected component
        max_fg_mask = _find_max_component(fg_mask)

        return max_fg_mask


class PredictForegroundV2(object):
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

    def __call__(
        self, image: np.ndarray, is_gray: bool = False, return_prob: bool = False
    ) -> np.ndarray:

        with torch.no_grad():
            in_image = (
                torch.tensor(image)[None, ...].permute(0, 3, 1, 2).float() / 255.0
            )
            print(in_image.shape)
            in_image = in_image.to(self._model_device)
            logits = self._model(in_image)[-1]
            is_fg_prob = logits.sigmoid().squeeze(0).cpu().numpy()

        if return_prob:
            return is_fg_prob
        fg_mask = is_fg_prob >= 0.5

        # get the largest connected component
        # max_fg_mask = _find_max_component(fg_mask)

        return fg_mask


if __name__ == "__main__":
    from PIL import Image

    from ..utils.misc import resize_with_aspect_ratio
    from ..utils.arg_parser import get_parser

    parser = get_parser("dev")
    args = parser.parse_args()

    pil_im = Image.open(
        os.path.join(
            os.getcwd(),
            "data",
            # "test",
            # "IMG_0020-fg.png",
            "train_data",
            "scanned",
            "images",
            "C2980",
            "no6-1009_143406.tif",
            # "no6-1009_173735.tif",
            # "FreeTalk04",
            # "no6-1022_162808.tif",
        )
    ).convert("RGB")
    im = np.array(pil_im.copy(), dtype=np.uint8)[..., ::-1]  # RGB to BGR
    inverted_im = np.flip(im.copy(), 1)
    resized_im, _ = resize_with_aspect_ratio(inverted_im, target_size=(1024, 1024))
    # resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)

    # pred_fg = PredictForegroundV2(model_name="rm_bg_test_ft033.pth")
    # fg_mask = pred_fg(resized_im.copy())

    # fg_mask = fg_mask.squeeze(0)
    # masked_im = resized_im * fg_mask[..., None]

    # cv2.imwrite(os.path.join("/", "home", "daniel", "Desktop", "before.jpg"), masked_im)

    pred_fg = PredictForegroundV2(model_name="rm_bg_test_C2980_1.pth")
    fg_mask = pred_fg(resized_im.copy())

    fg_mask = fg_mask.squeeze(0)
    masked_im = resized_im * fg_mask[..., None]
    # cv2.imwrite(os.path.join("/", "home", "daniel", "Desktop", "after.jpg"), masked_im)

    cv2.namedWindow("masked image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("masked image", 1024, 1024)
    cv2.imshow("masked image", masked_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
