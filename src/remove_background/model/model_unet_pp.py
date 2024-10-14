from typing import Optional

from torch import nn
import torch
from torchvision import models as tv_models

from .model_unet import ContractBlock


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        contract_block: nn.Module,
        embed_dims: list[int],
        number_of_classes: int,
        *args,
        **kwargs
    ):
        # embed_dims: [32, 64, 128, 256, 512]
        super(UNetPlusPlus, self).__init__(*args, **kwargs)

        self._contract_blk = contract_block

        self._expand_blks = nn.ModuleList()
        for ind, (curr_lvl_ch, next_lvl_ch) in enumerate(
            zip(embed_dims[:-1], embed_dims[1:])
        ):
            self._expand_blks.append(
                nn.ModuleList(
                    [
                        DoubleConvBlock(
                            in_channels=curr_lvl_ch * ind + next_lvl_ch,
                            inter_channels=curr_lvl_ch,
                            out_channels=curr_lvl_ch,
                            activation_fn=[nn.ReLU(), None],
                        )
                        for ind in range(1, len(embed_dims) - ind)
                    ]
                )
            )

        # add one more class for dummy class (background) if the number of classes is greater than 1
        all_classes: int = number_of_classes + (1 if number_of_classes > 1 else 0)
        self._to_logits = ConvBlock(
            in_channels=embed_dims[-5],
            out_channels=all_classes,
            kernel_size=1,
            bias=False,
            padding=0,
        )

        self._upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, src):
        # permute the input tensor to (batch, channel, height, width)
        src = src.permute(0, 3, 1, 2)
        # execute contractive block
        src_list: list[torch.Tensor] = self._contract_blk(src)
        src_list: list[list[torch.Tensor]] = [[src] for src in src_list]

        # execute expansive block
        for pyramid_level in range(1, len(src_list)):
            for ind in range(len(src_list) - pyramid_level):
                src_list[ind].append(
                    self._expand_blks[ind][pyramid_level - 1](
                        # concatenate the output of the previous block and the upsampled feature map
                        torch.cat(
                            src_list[ind] + [self._upsample(src_list[ind + 1][-1])],
                            dim=1,
                        )
                    )
                )

        logits = self._to_logits(torch.stack(src_list[0][1:], dim=1).mean(dim=1))
        return logits.permute(0, 2, 3, 1)


class ModifiedUNetPlusPlus(nn.Module):
    def __init__(
        self,
        contract_block,
        embed_dims: list[int],
        number_of_classes: int,
        *args,
        **kwargs
    ):
        # embed_dims: [32, 64, 128, 256, 512]
        super(UNetPlusPlus, self).__init__(*args, **kwargs)

        self._contract_blk = contract_block
        self._cls_num = number_of_classes

        self._expand_blks = nn.ModuleList()
        for ind, (curr_lvl_ch, next_lvl_ch) in enumerate(
            zip(embed_dims[:-1], embed_dims[1:])
        ):
            self._expand_blks.append(
                nn.ModuleList(
                    [
                        DoubleConvBlock(
                            in_channels=curr_lvl_ch * ind + next_lvl_ch,
                            inter_channels=curr_lvl_ch,
                            out_channels=curr_lvl_ch,
                            activation_fn=[nn.ReLU(), None],
                            bias=False,
                        )
                        for ind in range(1, len(embed_dims) - ind)
                    ]
                )
            )

        self._to_logits = ConvBlock(
            in_channels=embed_dims[-5],
            out_channels=self._cls_num,
            kernel_size=1,
            bias=False,
            padding=0,
        )

        self._upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, src):
        # permute the input tensor to (batch, channel, height, width)
        src = src.permute(0, 3, 1, 2)
        # execute contractive block
        src_list: list[torch.Tensor] = self._contract_blk(src)
        src_list: list[list[torch.Tensor]] = [[src] for src in src_list]

        # execute expansive block
        for pyramid_level in range(1, len(src_list)):
            for ind in range(len(src_list) - pyramid_level):
                src_list[ind].append(
                    self._expand_blks[ind][pyramid_level - 1](
                        # concatenate the output of the previous block and the upsampled feature map
                        torch.cat(
                            src_list[ind] + [self._upsample(src_list[ind + 1][-1])],
                            dim=1,
                        )
                    )
                )

        logits = self._to_logits(torch.stack(src_list[0][1:], dim=1).mean(dim=1))
        return logits.permute(0, 2, 3, 1)


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
        **kwargs
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


def build_unetplusplus(number_of_classes: int = 20) -> UNetPlusPlus:
    embed_dims = [64, 128, 256, 512, 1024]

    return UNetPlusPlus(
        contract_block=ContractBlock(embed_dims=embed_dims),
        embed_dims=embed_dims,
        number_of_classes=number_of_classes,
    )


def build_modified_unetplusplus(number_of_classes: int = 20) -> ModifiedUNetPlusPlus:
    embed_dims = [32, 64, 128, 256, 512]

    backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)

    contract_blk = nn.ModuleList()

    return ModifiedUNetPlusPlus(
        contract_block=ContractBlock(embed_dims=embed_dims),
        embed_dims=embed_dims,
        number_of_classes=number_of_classes + 1,
    )


if __name__ == "__main__":
    model = build_unetplusplus()

    img = torch.rand(size=(2, 256, 256, 3))
    print(model(img).shape)
