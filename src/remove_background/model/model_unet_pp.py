from torch import nn
import torch

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


def build_model(
    number_of_classes: int = 20, embed_dims: list[int] = [32, 64, 128, 256, 512]
) -> Model:

    backbone = build_unetpp(in_channels=1, embed_dims=embed_dims)
    model = Model(backbone, number_of_classes)

    return model


if __name__ == "__main__":
    model = build_model()

    img = torch.rand(size=(2, 256, 256, 3))
    print(model(img).shape)
