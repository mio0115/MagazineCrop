import torch
from torch import nn

from ...model_templates.blocks.blocks import ConvBlock


class ContractBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, norm: str = "batch"):
        super(ContractBlock, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.conv2 = ConvBlock(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.relu = nn.ReLU()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, src):
        feats = self.conv1(src)
        feats = self.conv2(feats)
        identity = self.skip_conv(identity)

        feats = feats + identity
        out = self.relu(feats)
        out = self.downsample(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, norm: str = "batch"):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.conv2 = ConvBlock(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.relu = nn.ReLU()

    def forward(self, src):
        feats = self.conv1(src)
        feats = self.conv2(feats)
        identity = self.skip_conv(identity)

        feats = feats + identity
        out = self.relu(feats)

        return out


class ExpanseBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, norm: str = "batch"):
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.conv2 = ConvBlock(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
            activation_fn=nn.ReLU(),
            normalization=norm,
        )
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


class Model(nn.Module):
    def __init__(
        self,
        stage_channels: list[int] = [64, 128, 256, 512],
        bottleneck_channels: int = 1024,
        in_channels: int = 4,
        out_channels: int = 1,
        normalization: str = "batch",
        *args,
        **kwargs
    ):
        super(Model, self).__init__(*args, **kwargs)

        self._in_channels = in_channels
        self._out_channels = out_channels

        self.contract_stages = nn.ModuleList()
        in_chn = in_channels
        for out_chn in stage_channels:
            self.contract_stages.append(
                ContractBlock(
                    in_channels=in_chn,
                    inter_channels=out_chn,
                    out_channels=out_chn,
                    norm=normalization,
                )
            )

        self.bottleneck = Bottleneck(
            in_channels=stage_channels[-1],
            inter_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            norm=normalization,
        )
