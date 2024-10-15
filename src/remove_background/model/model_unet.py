import torch
from torch import nn

from .blocks import ContractBlock, ExpansiveBlock


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

    def forward(self, src):
        # permute the input tensor to (batch, channel, height, width)
        src = src.permute(0, 3, 1, 2)

        # execute contractive block
        src_list: list[torch.Tensor] = self._contract_blk(src)
        # execute expansive block
        src_mask: torch.Tensor = self._expansive_blk(src_list).squeeze(1)

        return src_mask


def build_unet(embed_dims: list[int] = [64, 128, 256, 512, 1024]) -> nn.Module:
    contract_blk = ContractBlock(embed_dims=embed_dims)
    expansive_blk = ExpansiveBlock(embed_dims=embed_dims[::-1])

    model = UNet(
        contract_block=contract_blk,
        expansive_block=expansive_blk,
    )

    return model


if __name__ == "__main__":
    model = build_unet()

    img = torch.rand(size=(2, 256, 256, 3))

    print(model(img).shape)
