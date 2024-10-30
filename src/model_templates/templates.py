import torch
from torch import nn
from torchvision import models as tv_models

from .blocks.blocks import DoubleConvBlock, ContractBlock, BalanceConvBlock


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
                        DoubleConvBlock(
                            in_channels=curr_lvl_ch * ind + next_lvl_ch,
                            inter_channels=curr_lvl_ch,
                            out_channels=curr_lvl_ch,
                            bias=False,
                            activation_fn=[nn.ReLU(), None],
                        )
                        for ind in range(1, len(embed_dims) - ind)
                    ]
                )
            )

        self._weights = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.out_channels = embed_dims[0]

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

        feature_maps = torch.stack(src_list[0][1:], dim=1)
        weighted_features = torch.sum(
            self._weights.softmax(-1).view(1, -1, 1, 1, 1) * feature_maps, dim=1
        )

        return weighted_features


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeaturePyramidNetwork, self).__init__(*args, **kwargs)

        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        self._resnet_prev_layer = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self._resnet_blks = nn.ModuleList(
            [backbone.layer2, backbone.layer3, backbone.layer4]
        )
        self._up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self._conv1x1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                ),
            ]
        )
        self._conv3x3 = nn.ModuleList(
            [
                BalanceConvBlock(256, 256),
                BalanceConvBlock(256, 256),
                BalanceConvBlock(256, 256),
            ]
        )

    def forward(self, src):
        # execute the previous layer
        src = self._resnet_prev_layer(src)

        # execute the resnet blocks
        src_list = []
        for resnet_blk, conv_blk in zip(self._resnet_blks, self._conv1x1):
            src = resnet_blk(src)
            src_list.append(conv_blk(src))
        c2, c3, c4 = src_list

        # execute the top-down pathway
        p4 = c4
        p3 = c3 + self._up_sample(p4)
        p2 = c2 + self._up_sample(p3)

        # execute the lateral connections
        p2 = self._conv3x3[0](p2)
        p3 = self._conv3x3[1](p3)
        p4 = self._conv3x3[2](p4)
        src_list = [p2, p3, p4]

        return src_list


def build_unetpp(
    in_channels: int = 3, embed_dims: list[int] = [32, 64, 128, 256, 512]
) -> UNetPlusPlus:
    contract_block = ContractBlock(input_dim=in_channels, embed_dims=embed_dims)
    model = UNetPlusPlus(contract_block, embed_dims)

    return model


def build_fpn() -> FeaturePyramidNetwork:
    model = FeaturePyramidNetwork()

    return model


if __name__ == "__main__":
    model = FeaturePyramidNetwork()

    t = torch.rand(2, 640, 640, 3)
    out = model(t)

    print(out[0].shape, out[1].shape, out[2].shape)
