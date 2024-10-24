import torch
from torch import nn
import torchvision as tv


class SimpleLineEstimator(nn.Module):
    def __init__(self, backbone: nn.Module, *args, **kwargs):
        super(SimpleLineEstimator, self).__init__(*args, **kwargs)

        self._backbone = backbone
        self._fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        features = self._backbone(src)
        logits = self._fc(features)

        return logits


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Backbone, self).__init__(*args, **kwargs)

        self._backbone = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
        self._backbone.fc = nn.Identity()

    def forward(self, src):
        return self._backbone(src)


def build_model():
    backbone = Backbone()
    model = SimpleLineEstimator(backbone)

    return model


if __name__ == "__main__":
    model = build_model()

    output = model(torch.rand(size=(1, 3, 224, 224)))

    print(output.shape)
