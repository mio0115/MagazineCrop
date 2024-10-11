from torch import nn
import torch

from .model_unet import ContractBlock


class UNetPlusPlus(nn.Module):
    def __init__(
        self, contract_block: nn.Module, embed_dims: list[int], *args, **kwargs
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
                        ConvBlock(
                            in_channels=curr_lvl_ch * ind + next_lvl_ch,
                            out_channels=curr_lvl_ch,
                            activation_fn=nn.ReLU(),
                        )
                        for ind in range(1, len(embed_dims) - ind)
                    ]
                )
            )

        self._to_logits = ConvBlock(
            in_channels=embed_dims[-5],
            out_channels=1,
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
        return logits.squeeze(1)


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


def build_unetplusplus() -> UNetPlusPlus:
    embed_dims = [32, 64, 128, 256, 512]

    return UNetPlusPlus(
        contract_block=ContractBlock(embed_dims=embed_dims),
        embed_dims=embed_dims,
    )


if __name__ == "__main__":
    model = build_unetplusplus()

    img = torch.rand(size=(2, 256, 256, 3))
    print(model(img).shape)
