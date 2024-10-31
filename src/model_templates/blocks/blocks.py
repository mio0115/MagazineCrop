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

        # output the probability of the pixel being foreground
        self._output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self._embed_dims[-1],
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

        self._conv_blocks = nn.ModuleList()
        # self._expansive_layers = nn.ModuleList()
        self._expand = nn.UpsamplingBilinear2d(scale_factor=2)

        for channels in self._embed_dims[1:]:
            self._conv_blocks.append(
                DoubleConvBlock(
                    in_channels=channels * 2,
                    inter_channels=channels,
                    out_channels=channels,
                )
            )

    def forward(self, records: list[torch.Tensor]) -> torch.Tensor:
        # reverse the records to match the order of the expansive layers
        records = records[::-1]

        x = records[0]
        for src, conv_blk in zip(records[1:], self._conv_blocks):
            x = self._expand(x)
            # concat outputs of the contractive block with outputs of previous expansive layer
            x = torch.cat([x, src], dim=1)
            x = conv_blk(x)

        # output the logits of the pixel being foreground
        # remember to apply the sigmoid function to get the probability
        logits = self._output_layer(x)

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

        channels = [in_channels, inter_channels, out_channels]

        self._conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ks,
                    padding=p,
                    stride=s,
                    bias=b,
                    activation_fn=a,
                )
                for in_channels, out_channels, ks, p, s, b, a in zip(
                    channels[:-1],
                    channels[1:],
                    kernel_size,
                    padding,
                    stride,
                    bias,
                    activation_fn,
                )
            ]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        for conv_block in self._conv_blocks:
            src = conv_block(src)

        return src


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

        hori_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self._init_to_sobel(hori_conv, "horizontal")
        self._hori_conv_blk = nn.Sequential(
            hori_conv,
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
        hori_conv = self._hori_conv_blk(src)
        vert_conv = self._vert_conv_blk(src)

        balance_weight = self._balance_weight.sigmoid()
        x = balance_weight * hori_conv + (1 - balance_weight) * vert_conv

        if self._activation_func is not None:
            x = self._activation_func(x)

        return x


if __name__ == "__main__":
    # test the blocks
    input_tensor = torch.randn(2, 3, 256, 256)

    output = BalanceConvBlock(3, 64)(input_tensor)
    print(output.shape)
