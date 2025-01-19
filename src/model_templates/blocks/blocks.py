from typing import Optional

import torch
from torch import nn


class ContractBlock(nn.Module):
    def __init__(self, input_dim: int, embed_dims: list[int], *args, **kwargs):
        super(ContractBlock, self).__init__(*args, **kwargs)

        self._embed_dims = [input_dim] + embed_dims  # (3, 64, 128, 256, 512, 1024)

        self._conv_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(
            self._embed_dims[:-1], self._embed_dims[1:]
        ):
            self._conv_blocks.append(
                DoubleConvBlock(
                    in_channels=in_channels,
                    inter_channels=out_channels,
                    out_channels=out_channels,
                    bias=False,
                    activation_fn=nn.ReLU(),
                )
            )

        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, src: torch.Tensor) -> list[torch.Tensor]:
        output = []

        x = src
        for blk in self._conv_blocks:
            # execute the convolution block
            x = blk(x)
            # save the output of each block for skip connection
            output.append(x)

            x = self._max_pool(x)

        return output


class ExpansiveBlock(nn.Module):
    def __init__(self, embed_dims: list[int], *args, **kwargs):
        super(ExpansiveBlock, self).__init__(*args, **kwargs)

        self._embed_dims = embed_dims  # [1024, 512, 256, 128, 64]

        self._conv_blocks = nn.ModuleList()
        for channels in self._embed_dims[1:]:
            self._conv_blocks.append(
                DoubleConvBlock(
                    in_channels=channels * 2,
                    inter_channels=channels,
                    out_channels=channels,
                )
            )

        self._upsample = nn.ModuleList()
        for prev_channels, channels in zip(self._embed_dims, self._embed_dims[1:]):
            upsample_blk = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(prev_channels, channels, kernel_size=1, bias=False),
            )
            self._upsample.append(upsample_blk)

    def forward(self, records: list[torch.Tensor]) -> torch.Tensor:
        # reverse the records to match the order of the expansive layers
        records = records[::-1]

        x = records[0]
        for src, conv_blk, upsample_blk in zip(
            records[1:], self._conv_blocks, self._upsample
        ):
            x = upsample_blk(x)
            # concat outputs of the contractive block with outputs of previous expansive layer
            x = torch.cat([x, src], dim=1)
            x = conv_blk(x)

        # output the logits of the pixel being foreground
        # remember to apply the sigmoid function to get the probability
        logits = x

        return logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int] = 3,
        padding: int | tuple[int] = 1,
        stride: int | tuple[int] = 1,
        bias: bool = True,
        activation_fn: nn.Module = None,
    ):
        super(ConvBlock, self).__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self._bn = nn.BatchNorm2d(out_channels)
        self._activation_fn = activation_fn

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self._conv(src)
        src = self._bn(src)
        if self._activation_fn is not None:
            src = self._activation_fn(src)

        return src


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int] = 3,
        padding: int | tuple[int] = 1,
        stride: int | tuple[int] = 1,
        bias: bool | list[bool] = True,
        activation_fn: Optional[nn.Module | list[nn.Module]] = None,
        *args,
        **kwargs,
    ):
        super(DoubleConvBlock, self).__init__(*args, **kwargs)

        # extend the parameters to list if they are not
        if isinstance(bias, bool):
            bias = [bias, bias]
        if isinstance(activation_fn, nn.Module) or activation_fn is None:
            activation_fn = [activation_fn, activation_fn]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(stride, int):
            stride = [stride, stride]

        self._conv_blocks = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=kernel_size[0],
                padding=padding[0],
                stride=stride[0],
                bias=bias[0],
                activation_fn=activation_fn[0],
            ),
            ConvBlock(
                in_channels=inter_channels,
                out_channels=out_channels,
                kernel_size=kernel_size[1],
                padding=padding[1],
                stride=stride[1],
                bias=bias[1],
                activation_fn=activation_fn[1],
            ),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self._conv_blocks(src)

        return x


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SqueezeExcitationBlock, self).__init__()

        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # squeeze
        x = self._pool(src)
        x = x.view(x.size(0), -1)

        # excitation
        x = self._fc(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return src * x


class BalanceConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_func: Optional[callable] = None,
        *args,
        **kwargs,
    ):
        super(BalanceConvBlock, self).__init__(*args, **kwargs)

        norm_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # self._init_to_sobel(hori_conv, "horizontal")
        self._norm_conv_blk = nn.Sequential(
            norm_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        vert_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self._init_to_sobel(vert_conv, "vertical")
        self._vert_conv_blk = nn.Sequential(
            vert_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._balance_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._activation_func = activation_func

    def _init_to_sobel(self, conv: nn.Conv2d, choice: str):
        if choice == "horizontal":
            sobel = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).expand(conv.out_channels, conv.in_channels, 3, 3)
        elif choice == "vertical":
            sobel = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).expand(conv.out_channels, conv.in_channels, 3, 3)
        else:
            raise ValueError("Invalid choice")

        conv.weight = nn.Parameter(sobel)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        norm_conv = self._norm_conv_blk(src)
        vert_conv = self._vert_conv_blk(src)

        balance_weight = self._balance_weight.sigmoid()
        x = balance_weight * norm_conv + (1 - balance_weight) * vert_conv

        if self._activation_func is not None:
            x = self._activation_func(x)

        return x


class ResidualBalanceConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_func: Optional[callable] = None,
        *args,
        **kwargs,
    ):
        super(ResidualBalanceConvBlock, self).__init__(*args, **kwargs)

        self._conv_block = BalanceConvBlock(in_channels, out_channels, activation_func)
        self._residual_conv = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self._conv_block(src) + self._residual_conv(src)


class ScaleAttention(nn.Module):
    def __init__(self, in_channels: int = 256):
        super(ScaleAttention, self).__init__()

        self._to_weights = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 3),
        )

    def forward(self, features) -> torch.Tensor:
        weights = self._to_weights(torch.cat(features, 1).permute(0, 2, 3, 1)).softmax(
            -1
        )
        # features[0]: (bs, c, h, w)
        # different weights for each feature map
        # for each feature map, same weights for each channel; different weights for each pixel
        bs, h, w, _ = weights.shape
        reduced_features = (
            features[0] * weights[..., 0].view(bs, 1, h, w)
            + features[1] * weights[..., 1].view(bs, 1, h, w)
            + features[2] * weights[..., 2].view(bs, 1, h, w)
        )

        return reduced_features


class SpatialScaleAttention(nn.Module):
    def __init__(self, in_channels: int = 256, factor: int = 3):
        super(SpatialScaleAttention, self).__init__()

        self._to_weights = nn.Conv2d(
            in_channels * factor, factor, kernel_size=1, bias=False
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        concat_features = torch.cat(features, 1)
        weights = self._to_weights(concat_features).softmax(1)

        weighted_features = []
        for idx, feature in enumerate(features):
            weighted_features.append(feature * weights[:, idx : idx + 1])

        reduced_features = torch.stack(weighted_features, 1).sum(1)

        return reduced_features


class ChannelScaleAttention(nn.Module):
    def __init__(self, in_channels: int = 256):
        super(ChannelScaleAttention, self).__init__()

        self._to_weights = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1),
        )

    def forward(self, features):
        bs, ch, h, w = features[0].shape
        concat_features = torch.cat(features, 1)
        weights = self._to_weights(concat_features).view(bs, 3, ch, h, w).softmax(1)

        reduced_features = (
            features[0] * weights[:, 0]
            + features[1] * weights[:, 1]
            + features[2] * weights[:, 2]
        )

        return reduced_features


class LineApproxBlock(nn.Module):
    """
    A block that approximates lines on the left and right edges of a scanned page,
    modifying the input tensor (src) based on those lines.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the DoubleConvBlock.
        embed_channels (List[int]): Sizes of hidden layers in the regression MLP.
        src_shape (Tuple[int, int]): The (width, height) shape for the source image.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1024,
        embed_channels: list[int] = (512, 128, 64),
        src_shape: tuple[int, int] = (640, 640),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 1) Convolutional feature extractor
        self._conv = DoubleConvBlock(
            in_channels=in_channels,
            inter_channels=256,
            out_channels=out_channels,
            kernel_size=((1, 3), (1, 3)),
            padding=((0, 1), (0, 1)),
            stride=1,
            activation_fn=nn.ReLU(),
            bias=False,
        )

        # 2) Global average pool to reduce spatial dimension
        self._gap = nn.AdaptiveAvgPool2d(1)

        # 3) Regression MLP
        embed_channels = [out_channels] + list(embed_channels)
        self._regression_head = nn.ModuleList()
        for prev_chn, chn in zip(embed_channels[:-1], embed_channels[1:]):
            self._regression_head.append(
                nn.Sequential(
                    nn.Linear(prev_chn, chn, bias=False),
                    nn.ReLU(),
                )
            )

        # 4) Final line-approx layer: outputs 6 parameters (3 for left edge, 3 for right edge)
        self._lines_approx = nn.Sequential(
            nn.Linear(embed_channels[-1] + 1, 6), nn.Sigmoid()  # +1 for edge_theta
        )

        # Register buffers for constant geometry references
        w, h = src_shape
        x_coords = torch.arange(w)
        y_coords = torch.arange(h)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")

        self.register_buffer(
            "_src_top_left", torch.tensor([0, 0], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "_src_bottom_left", torch.tensor([0, h], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "_src_top_right", torch.tensor([w, 0], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "_src_bottom_right", torch.tensor([w, h], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "_grid", torch.stack([grid_x, grid_y], dim=-1)[None, ...].flatten(1, 2)
        )
        self.register_buffer(
            "_increment", torch.tensor(5.0, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "_decrement", torch.tensor(5.0, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "_scale_factors", torch.tensor([w, h], dtype=torch.float32)
        )

    def convex_mask(self, grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Generate a binary mask for the convex hull defined by 4 points, using cross-product tests.

        Args:
            grid (Tensor): shape (batch_size, height*width, 2), pixel coordinates.
            points (Tensor): shape (batch_size, 4, 2), the polygon corners as (x, y).

        Returns:
            Tensor: A binary mask (bool) of shape (batch_size, height, width) with True for
            pixels inside the convex hull.
        """
        points = points.float()  # ensure float for cross-product
        bs, num_pts, _ = points.shape

        # Start with all True, then refine via cross-product checks
        flatten_mask = torch.ones(
            grid.shape[:-1], dtype=torch.bool, device=points.device
        )

        for i in range(num_pts):
            p1 = points[:, i]
            p2 = points[:, (i + 1) % num_pts]
            edge = p2 - p1
            to_pixel = grid - p1.view(bs, 1, -1)

            cross_product = (
                edge[:, 0] * to_pixel[..., 1] - edge[:, 1] * to_pixel[..., 0]
            )

            # Inside if cross_product <= 0 for all edges (assuming consistent winding)
            flatten_mask &= cross_product <= 0

        return flatten_mask

    def forward(
        self, src: torch.Tensor, edge_len: torch.Tensor, edge_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that:
          1) Extracts features from src via conv & MLP,
          2) Predicts line parameters (left & right edges),
          3) Constructs polygons based on those lines,
          4) Draws them on src by subtracting/adding increments.

        Args:
            src (Tensor): shape (batch_size, channels, height, width), the input image/feature.
            edge_len (Tensor): shape (batch_size,), length for each edge in the batch.
            edge_theta (Tensor): shape (batch_size,), angle in degrees for each edge in the batch.

        Returns:
            Tensor: Modified src after applying lines-based increments/decrements.
        """

        bs, _, h, w = src.shape

        # 1) Feature extraction
        x = self._conv(src)
        x = self._gap(x).view(bs, -1)

        # 2) Pass through regression MLP
        for layer in self._regression_head:
            x = layer(x)

        # 3) Lines approximation
        #   - We also feed in the normalized angle (edge_theta / 180).
        angle_input = edge_theta.view(-1, 1) / 180.0
        coords = self._lines_approx(torch.cat([x, angle_input], dim=1))

        # coords -> 6 params: left( x, y, ??? ) + right( x, y, ??? )
        left_side, right_side = torch.chunk(coords, chunks=2, dim=1)

        # Scale xy by image size
        left_xy = left_side[:, :2] * self._scale_factors.view(1, 2)
        right_xy = right_side[:, :2] * self._scale_factors.view(1, 2)

        # Convert angles from [0..180]
        left_theta = 180.0 - left_side[:, -1] * 180.0
        right_theta = 180.0 - right_side[:, -1] * 180.0

        # 4) Corner points for left, right
        top_left = left_xy
        bottom_left = left_xy + edge_len.unsqueeze(1) * torch.stack(
            [torch.cos(left_theta), torch.sin(left_theta)], dim=1
        )
        top_right = right_xy
        bottom_right = right_xy + edge_len.unsqueeze(1) * torch.stack(
            [torch.cos(right_theta), torch.sin(right_theta)], dim=1
        )

        # 5) Create masks and modify src
        mask_left = self._make_polygon_mask(
            top_left, bottom_left, self._src_bottom_left, self._src_top_left, *src.shape
        )
        mask_right = self._make_polygon_mask(
            top_right,
            self._src_top_right,
            self._src_bottom_right,
            bottom_right,
            *src.shape,
        )
        after_dec = src - (mask_left + mask_right) * self._decrement

        # mid points
        mid_top = (top_left + top_right) / 2.0
        mid_bottom = (bottom_left + bottom_right) / 2.0

        # mid-mid logic
        mid_mid = (mid_top + mid_bottom) / 2.0
        mid_top = (mid_top + mid_mid) / 2.0
        mid_bottom = (mid_bottom + mid_mid) / 2.0

        mask_left_2 = self._make_polygon_mask(
            mid_top, mid_bottom, self._src_bottom_left, self._src_top_left, *src.shape
        )
        mask_right_2 = self._make_polygon_mask(
            mid_top, self._src_top_right, self._src_bottom_right, mid_bottom, *src.shape
        )
        after_inc = after_dec + (mask_left_2 + mask_right_2) * self._increment

        return after_inc

    def _make_polygon_mask(
        self,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        pt3: torch.Tensor,
        pt4: torch.Tensor,
        bs: int,
        ch: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Create a float mask for a polygon given 4 corners. Re-shapes and broadcasts to (bs, ch, h, w).
        """
        poly_pts = torch.stack(
            [pt1, pt2, pt3.expand_as(pt2), pt4.expand_as(pt1)], dim=0
        ).permute(
            1, 0, 2
        )  # shape (bs, 4, 2)

        mask = self.convex_mask(self._grid, poly_pts)
        mask = mask.reshape(bs, h, w).unsqueeze(1).expand(-1, ch, -1, -1)
        return mask.float()


if __name__ == "__main__":
    # test the blocks
    input_tensor = torch.randn(2, 3, 256, 256)

    output = BalanceConvBlock(3, 64)(input_tensor)
    print(output.shape)
