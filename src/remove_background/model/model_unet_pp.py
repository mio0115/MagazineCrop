from torch import nn
import torch

from ...model_templates.blocks.blocks import DoubleConvBlock
from ...model_templates.templates import build_unetpp


class Model(nn.Module):
    def __init__(self, backbone, number_of_classes: int, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self._backbone = backbone

        # add one more class for dummy class (background) if the number of classes is greater than 1
        all_classes: int = number_of_classes + (1 if number_of_classes > 1 else 0)
        self._to_logits = nn.Conv2d(
            in_channels=self._backbone.out_channels,
            out_channels=all_classes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, src):
        features = self._backbone(src)

        logits = self._to_logits(features)
        return logits.permute(0, 2, 3, 1).contiguous()


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
            nn.BatchNorm2d(backbones[0].out_channels), nn.ELU()
        )

    def forward(self, src):
        logits = []

        x = self._warmup(src)
        for backbone in self._backbones:
            tmp_x = backbone(x)
            x = x + self._residual_blk(tmp_x)

            logits.append(self._to_logits(x).permute(0, 2, 3, 1).contiguous())

        return logits


def build_model(
    number_of_classes: int = 20, embed_dims: list[int] = [32, 64, 128, 256, 512]
) -> Model:

    backbone = build_unetpp(in_channels=1, embed_dims=embed_dims)
    model = Model(backbone, number_of_classes)

    return model


def build_iterative_model(embed_dims: list[int] = [32, 64, 128, 256, 512]):
    backbones = [
        build_unetpp(in_channels=32, embed_dims=embed_dims[:ind])
        for ind in range(len(embed_dims), 1, -1)
    ]
    model = IterativeModel(in_channels=1, backbones=backbones)

    return model


if __name__ == "__main__":
    model = build_iterative_model()

    img = torch.rand(size=(2, 1, 256, 256))
    outputs = model(img)

    for output in outputs:
        print(output.shape)
    # print(type(outputs), len(outputs))
