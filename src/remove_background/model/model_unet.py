import torch
from torch import nn


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


class ContractBlock(nn.Module):
    def __init__(self, embed_dims: list[int], *args, **kwargs):
        super(ContractBlock, self).__init__(*args, **kwargs)

        self._embed_dims = [3] + embed_dims  # (3, 64, 128, 256, 512, 1024)

        self._conv_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(
            self._embed_dims[:-1], self._embed_dims[1:]
        ):
            self._conv_blocks.append(
                DoubleConv(channels=[in_channels, out_channels, out_channels])
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
                DoubleConv(channels=[channels * 2, channels, channels])
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


class DoubleConv(nn.Module):
    def __init__(self, channels: list[int], *args, **kwargs) -> None:
        super(DoubleConv, self).__init__(*args, **kwargs)

        assert (
            len(channels) == 3
        ), f"Channels to build DoubleConv must be 3. But get {len(channels)} instead."

        self._conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=channels[2]),
            nn.ReLU(),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self._conv_block(src)


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
