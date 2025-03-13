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
                f"{model_name}.pth",
            ),
            weights_only=False,
        )
        self._model.to(self._model_device)

        self._verbose = verbose
        self._new_size = new_size

    def __call__(
        self,
        image: np.ndarray,
        is_gray: bool = False,
        return_prob: bool = False,
        return_logits_indices: list[int] = [-1],
    ) -> np.ndarray:

        with torch.no_grad():
            in_image = (
                torch.tensor(image)[None, ...].permute(0, 3, 1, 2).float() / 255.0
            )
            in_image = in_image.to(self._model_device)
            logits = self._model(in_image)["logits"]

            is_fg_prob = []
            for idx in return_logits_indices:
                is_fg_prob.append(logits[idx].sigmoid().squeeze(0).cpu().numpy())

        if return_prob:
            return is_fg_prob
        masks = []
        for mask in is_fg_prob:
            fg_mask = mask >= 0.5
            masks.append(fg_mask)

        # get the largest connected component
        # max_fg_mask = _find_max_component(fg_mask)

        return masks


if __name__ == "__main__":
    from PIL import Image
    import subprocess

    from ..utils.misc import resize_with_aspect_ratio
    from ..utils.arg_parser import get_parser

    def _to_pil(image: np.ndarray) -> Image:
        return Image.fromarray(image)

    parser = get_parser("dev")
    args = parser.parse_args()

    path_to_infer_dir = os.path.join(
        os.getcwd(), "data", "train_data", "scanned", "images", "C2980"
    )
    pred_fg = PredictForegroundV2(model_name="rm_bg_iter_C2980.pth")
    count = 0

    for img_name in os.listdir(path_to_infer_dir):
        if not img_name.endswith(".tif"):
            continue
        count += 1
        pil_im = Image.open(
            os.path.join(
                path_to_infer_dir,
                img_name,
            )
        ).convert("RGB")
        im = np.array(pil_im.copy(), dtype=np.uint8)[..., ::-1]  # RGB to BGR
        resized_im, _ = resize_with_aspect_ratio(im, target_size=(1024, 1024))
        # inverted_im = np.flip(resized_im, axis=1)
        resized_gray_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
        resized_gray_im = cv2.equalizeHist(resized_gray_im)
        input_im = np.concatenate([resized_im, resized_gray_im[..., None]], axis=-1)
        # resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)

        # pred_fg = PredictForegroundV2(model_name="rm_bg_test_ft033.pth")
        # fg_mask = pred_fg(resized_im.copy())

        # fg_mask = fg_mask.squeeze(0)
        # masked_im = resized_im * fg_mask[..., None]

        # cv2.imwrite(os.path.join("/", "home", "daniel", "Desktop", "before.jpg"), masked_im)

        masks = pred_fg(input_im, return_logits_indices=list(range(4)))

        resized_pil_im = _to_pil(resized_im)
        resized_pil_im.save(
            os.path.join("/", "home", "daniel", "Desktop", "before.jpg")
        )
        for idx, mask in enumerate(masks):
            fg_mask = mask.squeeze(0)
            masked_im = resized_im.copy() * fg_mask[..., None]

            masked_im = _to_pil(masked_im)
            masked_im.save(os.path.join("/", "home", "daniel", "Desktop", "after.jpg"))

            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.utils.output",
                    "--original-image",
                    "/home/daniel/Desktop/before.jpg",
                    "--processed-images",
                    "/home/daniel/Desktop/after.jpg",
                    "--output-directory",
                    "/home/daniel/Desktop/valid_images",
                    "--output-name",
                    f"{count}_{idx+1}",
                ]
            )

        # cv2.namedWindow("masked image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("masked image", 1024, 1024)
        # cv2.imshow("masked image", masked_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
