import torch
from torch import nn


class ModIOUMetric(nn.Module):
    """
    A transform that computes an approximate Intersection-over-Union (IoU) between
    two convex polygons (with 4 corners) by rasterizing onto a (height, width) grid.

    This metric is *not differentiable* because we are using boolean masks
    based on cross products, but it provides a straightforward IoU measure
    for 4-point polygons.

    Args:
        height (int): The height of the grid for rasterization.
        width (int): The width of the grid for rasterization.
        reduction (str): Specifies how to reduce the IoU across the batch:
                         'mean' | 'sum' | 'none'.
                         Defaults to 'mean'.
        background_label (int): The label or threshold considered as "outside"
                                polygon for cross-product checks. Often 0.
    """

    def __init__(
        self, height: int, width: int, reduction: str = "mean", *args, **kwargs
    ):
        super(ModIOUMetric, self).__init__(*args, **kwargs)
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Reduction '{reduction}' is not supported.")

        self._height = height
        self._width = width
        self._reduction = reduction

        x_coords = torch.arange(width, dtype=torch.float32)
        y_coords = torch.arange(height, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
        # Flatten and add batch dimension
        self.register_buffer(
            "_grid",
            torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).unsqueeze(0),
        )
        self.register_buffer(
            "_scale_factor",
            torch.tensor([width, height], dtype=torch.float32).expand(1, 1, 2),
        )

        # Reorder corners if needed: for example, if your data is (top_left, top_right, bottom_right, bottom_left),
        # and you need a different winding (like clockwise or CCW).
        # Adjust to match your dataset's corner ordering if desired.
        self._pts_order = [0, 2, 3, 1]

    def _make_polygon_mask(self, poly_pts: torch.Tensor) -> torch.Tensor:
        """
        Create a float mask for a polygon given 4 corners.
        """
        bs, *_ = poly_pts.shape

        mask = self._convex_mask(poly_pts)
        mask = mask.reshape(bs, self._height, self._width)
        return mask

    def _convex_mask(self, points: torch.Tensor) -> torch.Tensor:
        """
        Generate a binary mask for the convex hull defined by 4 points, using cross-product tests.

        Args:
            points (Tensor): shape (batch, 4, 2), the polygon corners as (x, y). Note that the 4 corners are clockwise ordered.

        Returns:
            Tensor: A binary mask (bool) of shape (batch, height*width) with True for
            pixels inside the convex hull.
        """
        points = points.float()  # ensure float for cross-product
        num_pts = points.shape[1]

        # Start with all True, then refine via cross-product checks
        mask = torch.ones(self._grid.shape[:-1], dtype=torch.bool, device=points.device)

        for i in range(num_pts):
            # p1, p2 => shape (batch, 1, 2)
            p1 = points[:, i].unsqueeze(1)
            p2 = points[:, (i + 1) % num_pts].unsqueeze(1)
            edge = p2 - p1  # shape (batch, 1, 2)

            # shape => (1, h*w, 2) - (batch, 1, 2) => broadcast => (batch, h*w, 2)
            to_pixel = self._grid - p1

            cross_product = (
                edge[..., 0] * to_pixel[..., 1] - edge[..., 1] * to_pixel[..., 0]
            )

            # Inside if cross_product >= 0 for all edges (assuming consistent winding)
            # if the corners are ordered clockwise, the mask will be True inside the polygon
            mask = mask & (cross_product >= 0)

        return mask

    @torch.no_grad()
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the intersection over union (IoU) for a predicted polygon and a target polygon.

        Args:
            outputs (Tensor): shape (batch, 4, 2), the predicted polygon corners as (x, y).
            targets (Tensor): shape (batch, 4, 2), the target polygon corners as (x, y).

        Returns:
            Tensor: scalar tensor if reduction is 'mean' or 'sum', otherwise tensor of shape (batch,).
        """

        outputs = outputs.float() * self._scale_factor
        targets = targets.float() * self._scale_factor

        pred_mask = self._make_polygon_mask(outputs[:, self._pts_order])
        target_mask = self._make_polygon_mask(
            targets[:, self._pts_order],
        )

        intersection = (pred_mask & target_mask).sum(dim=(1, 2))
        union = (pred_mask | target_mask).sum(dim=(1, 2))
        iou = intersection.float() / union.float().clamp_min(1e-6)

        if self._reduction == "mean":
            iou = iou.mean()
        elif self._reduction == "sum":
            iou = iou.sum()
        return iou


class IOUMetric(nn.Module):
    """
    A metric that computes Intersection-over-Union (IoU) between
    masks by rasterizing onto a (height, width) grid.

    Args:
        height (int): The height of the grid for rasterization.
        width (int): The width of the grid for rasterization.
        reduction (str): Specifies how to reduce the IoU across the batch:
                         'mean' | 'sum' | 'none'.
                         Defaults to 'mean'.
        threshold (float): Threshold
    """

    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "mean",
        *args,
        **kwargs,
    ):
        super(IOUMetric, self).__init__(*args, **kwargs)
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Reduction '{reduction}' is not supported.")

        self._reduction = reduction
        self._thresh = threshold

    @torch.no_grad()
    def forward(
        self, logits: list[torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the intersection over union (IoU) for a predicted polygon and a target polygon.

        Args:
            logits (list[Tensor]): shape (batch, H, W), the predicted polygon corners as (x, y).
            targets (Tensor): shape (batch, H, W), the target polygon corners as (x, y).

        Returns:
            Tensor: scalar tensor if reduction is 'mean' or 'sum', otherwise tensor of shape (batch,).
        """

        ious = []
        target_mask = targets >= 0.9
        for logit in logits:
            pred_mask = (logit.sigmoid() >= self._thresh).squeeze(1)

            intersection = (pred_mask & target_mask).sum(dim=(1, 2))
            union = (pred_mask | target_mask).sum(dim=(1, 2))
            iou = intersection.float() / union.float().clamp_min(1e-6)

            if self._reduction == "mean":
                iou = iou.mean()
            elif self._reduction == "sum":
                iou = iou.sum()
            ious.append(iou)

        ious = torch.stack(ious)
        if self._reduction == "mean":
            ious = ious.mean()
        elif self._reduction == "sum":
            ious = ious.sum()
        return ious


if __name__ == "__main__":
    iou_metric = IOUMetric()

    pred = [torch.rand((4, 1, 640, 640)) for _ in range(5)]
    target = torch.rand((640, 640)) > 0.5

    iou = iou_metric(pred, target)
    print(iou)
