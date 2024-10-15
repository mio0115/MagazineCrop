from typing import Optional

import torch
from torch import nn


class ContractBlock(nn.Module):
    def __init__(self, embed_dims: list[int], *args, **kwargs):
        super(ContractBlock, self).__init__(*args, **kwargs)

        self._embed_dims = [3] + embed_dims  # (3, 64, 128, 256, 512, 1024)

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
