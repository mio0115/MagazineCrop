from typing import Optional

from torch import nn
import torch
import torch.utils.checkpoint as checkpoint

from ...model_templates.blocks.blocks import DoubleConvBlock, LineApproxBlock
from ...model_templates.templates import build_unetpp


class Backbone(nn.Module):
    def __init__(self, in_channels: int, backbones, *args, **kwargs):
        super(Backbone, self).__init__(*args, **kwargs)

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

    def forward(self, src):
        logits = []

        x = self._warmup(src)

        x = self._backbones[0](x)
        logits.append(self._to_logits(x))

        return logits


class Model(nn.Module):
    def __init__(
        self, backbone, src_shape: tuple[int, int] = (640, 640), *args, **kwargs
    ):
        super(Model, self).__init__(*args, **kwargs)

        self._backbone = backbone

        self._line_approx_block = LineApproxBlock(src_shape=src_shape)

    def forward(self, src, edge_len, edge_theta):
        logits = self._backbone(src)[0]
        return self._line_approx_block(logits, edge_len, edge_theta)


class InterModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InterModel, self).__init__(*args, **kwargs)

        self._line_approx_block = LineApproxBlock()

    def forward(self, src, edge_len, edge_theta):
        return self._line_approx_block(src, edge_len, edge_theta)


class CheckpointedModel(nn.Module):
    def __init__(self, model):
        super(CheckpointedModel, self).__init__()

        self._model = model

    def forward(self, *args):
        return checkpoint.checkpoint(self._model, *args)


def build_backbone(
    embed_dims: list[int] = [32, 64, 128, 256, 512],
    resume: bool = False,
    path_to_ckpt: Optional[str] = None,
):
    model = Backbone(
        in_channels=4, backbones=[build_unetpp(in_channels=32, embed_dims=embed_dims)]
    )

    if resume:
        model.load_state_dict(torch.load(path_to_ckpt, weights_only=True))

    return model


def build_model(path_to_backbone_ckpt: str = None):
    backbone = build_backbone(resume=True, path_to_ckpt=path_to_backbone_ckpt)
    model = Model(backbone)

    return model


if __name__ == "__main__":
    import os

    backbone = build_backbone(
        resume=True,
        path_to_ckpt=os.path.join(
            os.getcwd(),
            "src",
            "remove_background",
            "checkpoints",
            "rm_bg_iter_C2980_part_weights.pth",
        ),
    )
    model = Model(backbone)

    img = torch.rand(size=(2, 4, 640, 640))
    outputs = model(
        img,
        edge_len=torch.tensor([100.0, 105.0], dtype=torch.float32),
        edge_theta=torch.tensor([95.0, 89.0], dtype=torch.float32),
    )

    for output in outputs:
        print(output.shape)
