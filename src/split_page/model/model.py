import torch
from torch import nn
import torchvision as tv

from ...model_templates import templates


class SimpleLineEstimator(nn.Module):
    def __init__(self, backbone: nn.Module, *args, **kwargs):
        super(SimpleLineEstimator, self).__init__(*args, **kwargs)

        self._backbone = backbone
        self._avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self._x_reg_head = nn.Sequential(
            nn.Linear(32, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self._theta_reg_head = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # origin shape of src: (batch_size, channel, height, width)
        # permute the input tensor to (batch, height, width, channel)
        src = src.permute(0, 2, 3, 1)
        features = self._backbone(src)

        # reduce the spatial dimension of the feature map from (batch, channel, height, width) to (batch, channel)
        reduced_features = self._avg_pool(features).squeeze()

        x_logits = self._x_reg_head(reduced_features)
        theta_logits = self._theta_reg_head(reduced_features)

        return torch.cat([x_logits, theta_logits], dim=-1)


class Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Backbone, self).__init__(*args, **kwargs)

        self._backbone = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)
        self._backbone.layer4 = nn.Identity()
        self._backbone.fc = nn.Identity()

    def forward(self, src):
        return self._backbone(src)


def build_model():
    backbone = templates.build_unetpp(in_channels=3, embed_dims=[32, 64, 128, 256, 512])
    model = SimpleLineEstimator(backbone)

    return model


if __name__ == "__main__":
    model = build_model()

    output = model(torch.rand(size=(1, 3, 224, 224)))

    print(output.shape)
