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
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1024,
        embed_channels: list[int] = [512, 128, 64],
        *args,
        **kwargs,
    ):
        super(LineApproxBlock, self).__init__(*args, **kwargs)

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
        self._gap = nn.AdaptiveAvgPool2d(1)

        self._regression_head = nn.ModuleList()
        embed_channels = [out_channels] + embed_channels
        for prev_chn, chn in zip(embed_channels[:-1], embed_channels[1:]):
            self._regression_head.append(
                nn.Sequential(
                    nn.Linear(prev_chn, chn, bias=False),
                    nn.LayerNorm(chn),
                    nn.ReLU(),
                )
            )
        self._left_reg_head = nn.Sequential(
            nn.Linear(embed_channels[-1] + 1, 3), nn.Sigmoid()
        )
        self._right_reg_head = nn.Sequential(
            nn.Linear(embed_channels[-1] + 1, 3), nn.Sigmoid()
        )
        self._increment = nn.Parameter(
            torch.full((1,), 5, dtype=torch.float32), requires_grad=True
        )
        self._decrement = nn.Parameter(
            torch.full((1,), 5, dtype=torch.float32), requires_grad=True
        )

    import torch

    def convex_mask(self, grid, points):
        """
        Generate a binary mask for the convex hull defined by four points.

        Args:
            grid (torch.Tensor): Tensor of shape (batch_size, height * width, 2) containing the pixel coordinates.
            points (torch.Tensor): Tensor of shape (batch_size, 4, 2) containing the four points as (x, y).

        Returns:
            torch.Tensor: A binary mask of shape (batch_size, height, width) with 1s for pixels inside the convex hull.
        """

        points = points.float()  # Ensure float for computations

        bs, num_pts, _ = points.shape

        # Check if each pixel is inside the convex hull using cross-product tests
        flatten_mask = torch.ones(
            grid.shape[:-1], dtype=torch.bool, device=points.device
        )

        for i in range(num_pts):
            p1 = points[:, i]
            p2 = points[:, (i + 1) % num_pts]  # Next point
            edge = p2 - p1
            to_pixel = grid - p1.view(bs, 1, -1)
            cross_product = (
                edge[:, 0] * to_pixel[..., 1] - edge[:, 1] * to_pixel[..., 0]
            )
            flatten_mask &= (
                cross_product <= 0
            )  # Pixel is inside if all cross-products are positive

        return flatten_mask

    def forward(
        self, src: torch.Tensor, edge_len: torch.Tensor, edge_theta: torch.Tensor
    ) -> torch.Tensor:
        bs, ch, h, w = src.shape
        x = self._conv(src)

        x = self._gap(x).view(bs, -1)
        for layer in self._regression_head:
            x = layer(x)

        edge_theta = edge_theta.view(-1, 1) / 180
        left_side = self._left_reg_head(torch.cat([x, edge_theta], 1))
        right_side = self._right_reg_head(torch.cat([x, edge_theta], 1))

        scale_factors = torch.tensor(
            [w, h], device=src.device, dtype=torch.float32
        ).view(1, 2)

        left_xy = left_side[:, :2] * scale_factors
        right_xy = right_side[:, :2] * scale_factors

        left_theta = 180 - left_side[:, -1] * 180
        right_theta = 180 - right_side[:, -1] * 180

        # We then find the four corners of the box
        top_left = left_xy
        bottom_left = left_xy + edge_len * torch.stack(
            [torch.cos(left_theta), torch.sin(left_theta)], 1
        )
        top_right = right_xy
        bottom_right = right_xy + edge_len * torch.stack(
            [torch.cos(right_theta), torch.sin(right_theta)], 1
        )

        src_top_left = (
            torch.tensor([0, 0], device=src.device, dtype=torch.float32)
            .view(1, -1)
            .expand_as(top_left)
        )
        src_bottom_left = (
            torch.tensor([0, h], device=src.device, dtype=torch.float32)
            .view(1, -1)
            .expand_as(bottom_left)
        )
        src_top_right = (
            torch.tensor([w, 0], device=src.device, dtype=torch.float32)
            .view(1, -1)
            .expand_as(top_right)
        )
        src_bottom_right = (
            torch.tensor([w, h], device=src.device, dtype=torch.float32)
            .view(1, -1)
            .expand_as(bottom_right)
        )

        x_coords = torch.arange(w)
        y_coords = torch.arange(h)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
        grid = (
            torch.stack([grid_x, grid_y], dim=-1)
            .to(src.device)[None, ...]
            .flatten(1, 2)
        )

        mask = (
            (
                self.convex_mask(
                    grid,
                    torch.stack(
                        [top_left, bottom_left, src_bottom_left, src_top_left], 0
                    ).permute(1, 0, 2),
                )
            )
            .reshape(bs, h, w)
            .unsqueeze(1)
            .expand(-1, ch, -1, -1)
        )
        src[mask] -= self._decrement.relu()

        mask = (
            (
                self.convex_mask(
                    grid,
                    torch.stack(
                        [top_right, src_top_right, src_bottom_right, bottom_right], 0
                    ).permute(1, 0, 2),
                )
            )
            .reshape(bs, h, w)
            .unsqueeze(1)
            .expand(-1, ch, -1, -1)
        )
        src[mask] -= self._decrement.relu()

        mid_top = (top_left + top_right) / 2
        mid_bottom = (bottom_left + bottom_right) / 2

        mid_mid = (mid_top + mid_bottom) / 2
        mid_top = (mid_top + mid_mid) / 2
        mid_bottom = (mid_bottom + mid_mid) / 2

        mask = (
            (
                self.convex_mask(
                    grid,
                    torch.stack(
                        [mid_top, mid_bottom, src_bottom_left, src_top_left], 0
                    ).permute(1, 0, 2),
                )
            )
            .reshape(bs, h, w)
            .unsqueeze(1)
            .expand(-1, ch, -1, -1)
        )
        src[mask] += self._increment.relu()

        mask = (
            (
                self.convex_mask(
                    grid,
                    torch.stack(
                        [mid_top, src_top_right, src_bottom_right, mid_bottom], 0
                    ).permute(1, 0, 2),
                )
            )
            .reshape(bs, h, w)
            .unsqueeze(1)
            .expand(-1, ch, -1, -1)
        )
        src[mask] += self._increment.relu()

        return src


if __name__ == "__main__":
    # test the blocks
    input_tensor = torch.randn(2, 3, 256, 256)

    output = BalanceConvBlock(3, 64)(input_tensor)
    print(output.shape)
