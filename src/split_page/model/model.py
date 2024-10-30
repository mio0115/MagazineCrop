import torch
from torch import nn
import torchvision as tv

from ...model_templates import templates
from ...model_templates.blocks.blocks import BalanceConvBlock


class SimpleLineEstimator(nn.Module):
    def __init__(self, backbone: nn.Module, *args, **kwargs):
        super(SimpleLineEstimator, self).__init__(*args, **kwargs)

        self._backbone = backbone
        self._contract_small = nn.Sequential(
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256, activation_func=nn.LeakyReLU()),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._contract_medium = nn.Sequential(
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256, activation_func=nn.LeakyReLU()),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._contract_large = nn.Sequential(
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
            BalanceConvBlock(256, 256, activation_func=nn.LeakyReLU()),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self._weights = nn.Parameter(torch.ones((256, 3), dtype=torch.float32))
        self._x_reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        self._theta_reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        features_list = self._backbone(src)

        small_features = self._contract_small(features_list[0]).flatten(1)
        medium_features = self._contract_medium(features_list[1]).flatten(1)
        large_features = self._contract_large(features_list[2]).flatten(1)

        features = torch.stack(
            [small_features, medium_features, large_features], dim=-1
        )

        combined_features = torch.sum(
            features * self._weights[None, ...].softmax(dim=-1), dim=-1
        )

        x_logits = self._x_reg_head(combined_features)
        theta_logits = self._theta_reg_head(combined_features)

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
    backbone = templates.build_fpn()
    model = SimpleLineEstimator(backbone)

    return model


if __name__ == "__main__":
    model = build_model()

    output = model(torch.rand(size=(2, 3, 640, 640)))

    print(output.shape)
