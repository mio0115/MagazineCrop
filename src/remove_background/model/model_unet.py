import torch
from torch import nn

from ...model_templates.blocks.blocks import (
    ContractBlock,
    ExpansiveBlock,
    DoubleConvBlock,
)


class UNet(nn.Module):
    def __init__(
        self,
        contract_block: nn.Module,
        expansive_block: nn.Module,
        *args,
        **kwargs,
    ):
        super(UNet, self).__init__(*args, **kwargs)

        self._contract_blk = contract_block
        self._expansive_blk = expansive_block
        self.out_channels = expansive_block._embed_dims[-1]

    def forward(self, src):
        # execute contractive block
        src_list: list[torch.Tensor] = self._contract_blk(src)
        # execute expansive block
        src_mask: torch.Tensor = self._expansive_blk(src_list).squeeze(1)

        return src_mask


class IterativeModel(nn.Module):
    def __init__(self, in_channels: int, backbones, *args, **kwargs):
        super(IterativeModel, self).__init__(*args, **kwargs)

        self._warmup = DoubleConvBlock(
            in_channels=in_channels,
            inter_channels=backbones[0].out_channels,
            out_channels=backbones[0].out_channels,
            bias=False,
            activation_fn=nn.ReLU(),
        )

        self._backbones = nn.ModuleList(backbones)

        self._to_logits = nn.Conv2d(
            in_channels=backbones[0].out_channels,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        self._residual_blk = nn.Sequential(
            nn.BatchNorm2d(backbones[0].out_channels), nn.ReLU()
        )

    def forward(self, src):
        logits = []

        x = self._warmup(src)
        x = self._backbones[0](x)
        logits.append(self._to_logits(x).permute(0, 2, 3, 1).contiguous())

        for backbone in self._backbones[1:]:
            tmp_x = backbone(x)
            x = x + self._residual_blk(tmp_x)

            logits.append(self._to_logits(x).permute(0, 2, 3, 1).contiguous())

        return logits


def build_unet(
    in_channels, embed_dims: list[int] = [64, 128, 256, 512, 1024]
) -> nn.Module:
    contract_blk = ContractBlock(input_dim=in_channels, embed_dims=embed_dims)
    expansive_blk = ExpansiveBlock(embed_dims=embed_dims[::-1])

    model = UNet(
        contract_block=contract_blk,
        expansive_block=expansive_blk,
    )

    return model


def build_iterative_model(
    embed_dims: list[int] = [32, 64, 128, 256, 512], num_iter: int = 5
):
    backbones = [
        build_unet(in_channels=32, embed_dims=embed_dims) for _ in range(num_iter)
    ]
    model = IterativeModel(in_channels=3, backbones=backbones)

    return model


if __name__ == "__main__":
    model = build_iterative_model(num_iter=4)

    img = torch.rand(size=(2, 3, 1024, 1024))

    list_logits = model(img)
    for logits in list_logits:
        print(logits.shape)
