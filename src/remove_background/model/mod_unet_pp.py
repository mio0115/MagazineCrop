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
        self.out_channels = backbones[0].out_channels

    def forward(self, src):
        raws = []

        x = self._warmup(src)

        x = self._backbones[0](x)
        raws.append(x)

        return raws


class Model(nn.Module):
    def __init__(
        self, backbone, src_shape: tuple[int, int] = (640, 640), *args, **kwargs
    ):
        super(Model, self).__init__(*args, **kwargs)

        self._backbone = backbone
        self._to_logits = nn.Conv2d(
            in_channels=self._backbone.out_channels,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )

        self._line_approx_block = LineApproxBlock(
            in_channels=self._backbone.out_channels, src_shape=src_shape
        )

    def forward(
        self, src: torch.Tensor, edge_length: torch.Tensor, edge_theta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raws = self._backbone(src)[0]
        coords = self._line_approx_block(raws, edge_length, edge_theta)
        logits = self._to_logits(raws)
        outputs = {"logits": logits, "coords": coords}

        return outputs


class InterModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InterModel, self).__init__(*args, **kwargs)

        self._line_approx_block = LineApproxBlock()

    def forward(self, src, edge_len, edge_theta) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self._line_approx_block(src, edge_len, edge_theta)

        outputs = {"logits": src, "coords": coords}
        return outputs


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
        state_dict = torch.load(path_to_ckpt, weights_only=True)
        model.load_state_dict(state_dict)

    return model


def build_model(
    path_to_ckpt: str = None,
    src_shape: tuple[int, int] = (640, 640),
    backbone_name: str = "rm_bg_iter_C2980_weights.pth",
    to_logits_name: str = "rm_bg_to_logits_C2980_weights.pth",
):
    backbone = build_backbone(
        resume=True, path_to_ckpt=os.path.join(path_to_ckpt, backbone_name)
    )
    model = Model(backbone, src_shape=src_shape)

    to_logits_state = torch.load(
        os.path.join(path_to_ckpt, to_logits_name), weights_only=True
    )
    model._to_logits.load_state_dict(to_logits_state)

    return model


if __name__ == "__main__":
    import os

    model = build_model(
        path_to_ckpt=os.path.join(
            os.getcwd(), "src", "remove_background", "checkpoints"
        )
    )
    for key in model.state_dict().keys():
        print(key)

    img = torch.rand(size=(1, 4, 640, 640))
    outputs = model(
        img,
        edge_len=torch.tensor([100.0, 105.0], dtype=torch.float32),
        edge_theta=torch.tensor([95.0, 89.0], dtype=torch.float32),
    )

    for output in outputs:
        print(output.shape)
