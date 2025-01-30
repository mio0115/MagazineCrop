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


class PredictForegroundV2(object):
    def __init__(
        self,
        device: str = "cuda",
        split: bool = False,
        backbone: torch.nn.Module = None,
        model: torch.nn.Module = None,
        verbose: int = 0,
        new_size: tuple[int] = (1024, 1024),
    ):
        self._model_device = device
        self._backbone = backbone
        self._model = model
        self._split = split

        if backbone is not None:
            self._backbone.to(self._model_device)
        self._model.to(self._model_device)

        self._verbose = verbose
        self._new_size = new_size

    def __call__(
        self,
        image: np.ndarray,
        edge_len: float,
        edge_theta: float,
        is_gray: bool = False,
        return_prob: bool = False,
        return_logits_indices: list[int] = [3],
    ) -> np.ndarray:

        with torch.no_grad():
            in_image = (
                torch.tensor(image)[None, ...].permute(0, 3, 1, 2).float() / 255.0
            )
            edge_len = torch.tensor([edge_len]).to(self._model_device)
            edge_theta = torch.tensor([edge_theta]).to(self._model_device)

            in_image = in_image.to(self._model_device)
            if self._split:
                logits = self._backbone(in_image)[0]
                logits = self._model(logits, edge_len=edge_len, edge_theta=edge_theta)
            else:
                logits = self._model(in_image, edge_len=edge_len, edge_theta=edge_theta)
            is_fg_prob = []
            for idx in return_logits_indices:
                is_fg_prob.append(logits[idx].sigmoid().cpu().numpy())

        if return_prob:
            return is_fg_prob
        masks = []
        for mask in is_fg_prob:
            fg_mask = mask >= 0.5
            masks.append(fg_mask)

        # get the largest connected component
        # max_fg_mask = _find_max_component(fg_mask)

        return masks


class PredictForegroundV3(object):
    def __init__(
        self,
        device: str = "cuda",
        model: torch.nn.Module = None,
        verbose: int = 0,
        new_size: tuple[int] = (1024, 1024),
    ):
        self._model_device = device
        self._model = model

        self._model.to(self._model_device)

        self._verbose = verbose
        self._new_size = new_size

        w, h = new_size
        x_coords = torch.arange(w)
        y_coords = torch.arange(h)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")

        self._src_top_left = torch.tensor([0, 0], dtype=torch.float32)
        self._src_bottom_left = torch.tensor([0, h], dtype=torch.float32)
        self._src_top_right = torch.tensor([w, 0], dtype=torch.float32)
        self._src_bottom_right = torch.tensor([w, h], dtype=torch.float32)

        self._grid = torch.stack([grid_x, grid_y], -1).flatten(end_dim=1)
        self._constant = 100

    def _make_polygon_mask(
        self,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        pt3: torch.Tensor,
        pt4: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Create a float mask for a polygon given 4 corners. Re-shapes and broadcasts to (ch, h, w).
        """
        ch = 1
        poly_pts = torch.stack([pt1, pt2, pt3, pt4], dim=0)  # shape (4, 2)

        mask = self.convex_mask(self._grid, poly_pts)
        mask = mask.reshape(h, w).unsqueeze(0).expand(ch, -1, -1)
        return mask.float()

    def convex_mask(self, grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Generate a binary mask for the convex hull defined by 4 points, using cross-product tests.

        Args:
            grid (Tensor): shape (height*width, 2), pixel coordinates.
            points (Tensor): shape (4, 2), the polygon corners as (x, y).

        Returns:
            Tensor: A binary mask (bool) of shape (height*width) with True for
            pixels inside the convex hull.
        """
        points = points.float()  # ensure float for cross-product
        num_pts, _ = points.shape

        # Start with all True, then refine via cross-product checks
        mask = torch.ones(grid.shape[:-1], dtype=torch.bool, device=points.device)

        for i in range(num_pts):
            p1 = points[i]
            p2 = points[(i + 1) % num_pts]
            edge = p2 - p1
            to_pixel = grid - p1.view(1, -1)

            cross_product = edge[0] * to_pixel[:, 1] - edge[1] * to_pixel[:, 0]

            # Inside if cross_product >= 0 for all edges (assuming consistent winding)
            mask = mask & (cross_product >= 0)

        return mask

    def __call__(
        self,
        image: np.ndarray,
        edge_len: float,
        edge_theta: float,
        return_prob: bool = False,
    ) -> np.ndarray:

        with torch.no_grad():
            in_image = (
                torch.tensor(image)[None, ...].permute(0, 3, 1, 2).float() / 255.0
            )
            edge_len = torch.tensor([edge_len], dtype=torch.float32).to(
                self._model_device
            )
            edge_theta = torch.tensor([edge_theta], dtype=torch.float32).to(
                self._model_device
            )

            in_image = in_image.to(self._model_device)
            batch_outputs = self._model(
                in_image, edge_length=edge_len, edge_theta=edge_theta
            )

            outputs = {}
            for key, value in batch_outputs.items():
                outputs[key] = value.squeeze(0).cpu()

            outputs["coords"][:, 0] *= self._new_size[0]
            outputs["coords"][:, 1] *= self._new_size[1]

            top_left, bottom_left, top_right, bottom_right = outputs["coords"]
            mask_left = self._make_polygon_mask(
                top_left,
                bottom_left,
                self._src_bottom_left,
                self._src_top_left,
                *self._new_size,
            )
            mask_right = self._make_polygon_mask(
                top_right,
                self._src_top_right,
                self._src_bottom_right,
                bottom_right,
                *self._new_size,
            )
            # print(mask_left.sum(), mask_right.sum())
            outputs["logits"] = (
                outputs["logits"] - (mask_left + mask_right) * self._constant
            )

            # mid_top = (top_left + top_right) / 2.0
            # mid_bottom = (bottom_left + bottom_right) / 2.0

            # mid_mid = (mid_top + mid_bottom) / 2.0
            # mid_top = (mid_top + mid_mid) / 2.0
            # mid_bottom = (mid_bottom + mid_mid) / 2.0

            # mask_left_2 = self._make_polygon_mask(
            #     mid_top,
            #     mid_bottom,
            #     self._src_bottom_left.expand(bs, -1),
            #     self._src_top_left.expand(bs, -1),
            #     *src.shape,
            # )
            # mask_right_2 = self._make_polygon_mask(
            #     mid_top,
            #     self._src_top_right.expand(bs, -1),
            #     self._src_bottom_right.expand(bs, -1),
            #     mid_bottom,
            #     *src.shape,
            # )
            # after_inc = after_dec + (mask_left_2 + mask_right_2) * self._increment

            is_fg_prob = [outputs["logits"].sigmoid().cpu().numpy()]

        if return_prob:
            return is_fg_prob
        masks = []
        for mask in is_fg_prob:
            fg_mask = mask >= 0.5
            masks.append(fg_mask)

        # get the largest connected component
        # max_fg_mask = _find_max_component(fg_mask)

        return masks, (top_left, bottom_left, top_right, bottom_right)


if __name__ == "__main__":
    from PIL import Image
    import subprocess
    import json

    from ..utils.misc import resize_with_aspect_ratio
    from ..utils.arg_parser import get_parser

    def _to_pil(image: np.ndarray) -> Image:
        return Image.fromarray(image)

    parser = get_parser("dev")
    args = parser.parse_args()

    path_to_infer_dir = os.path.join(
        os.getcwd(), "data", "train_data", "scanned", "images", args.image_dir
    )
    model = torch.load(args.rm_bg_model, weights_only=False)

    pred_fg = PredictForegroundV3(
        model=model, new_size=(args.edge_size, args.edge_size)
    )

    with open(
        os.path.join(
            os.getcwd(),
            "data",
            "train_data",
            "scanned",
            "annotations",
            f"edge_annotations_{args.edge_size}.json",
        ),
        "r",
    ) as annotaion_file:
        edge_annotations = json.load(annotaion_file)

    edge_info = edge_annotations[args.image_dir]
    count = 0

    path_to_tmp = os.path.join("/", "tmp", "image-output")

    if not os.path.exists(path_to_tmp):
        subprocess.run(["mkdir", path_to_tmp])

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
        resized_im, _ = resize_with_aspect_ratio(
            im, target_size=(args.edge_size, args.edge_size)
        )
        resized_gray_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
        resized_gray_im = cv2.equalizeHist(resized_gray_im)
        input_im = np.concatenate([resized_im, resized_gray_im[..., None]], axis=-1)

        edge_len = edge_info["edge_length"]
        edge_theta = edge_info["theta"]
        edge_theta = np.deg2rad(edge_theta) / np.pi

        masks, coords = pred_fg(
            input_im,
            edge_len=edge_len,
            edge_theta=edge_theta,
        )

        resized_pil_im = _to_pil(resized_im)
        resized_pil_im.save(os.path.join(path_to_tmp, "before.jpg"))
        for idx, mask in enumerate(masks):
            fg_mask = mask.squeeze(0)

            masked_im = resized_im.copy() * fg_mask[..., None]
            cv2.line(
                masked_im,
                pt1=coords[0].int().numpy(),
                pt2=coords[1].int().numpy(),
                color=(0, 0, 255),
                thickness=3,
            )
            cv2.line(
                masked_im,
                pt1=coords[2].int().numpy(),
                pt2=coords[3].int().numpy(),
                color=(0, 255, 0),
                thickness=3,
            )

            masked_im = _to_pil(masked_im)
            masked_im.save(os.path.join(path_to_tmp, "after.jpg"))

            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.utils.output",
                    "--original-image",
                    os.path.join(path_to_tmp, "before.jpg"),
                    "--processed-images",
                    os.path.join(path_to_tmp, "after.jpg"),
                    "--output-directory",
                    args.output_dir,
                    "--output-name",
                    f"{count}",
                ]
            )
