import torch
from torch import nn
import torchvision as tv

from ...model_templates import templates
from ...model_templates.blocks.blocks import (
    BalanceConvBlock,
    ScaleAttention,
    SpatialScaleAttention,
    ChannelScaleAttention,
)


class SimpleLineEstimator(nn.Module):
    def __init__(self, backbone: nn.Module, *args, **kwargs):
        super(SimpleLineEstimator, self).__init__(*args, **kwargs)

        self._backbone = backbone

        self._for_small_scale = nn.Sequential(
            BalanceConvBlock(256, 512), BalanceConvBlock(512, 256)
        )
        self._for_medium_scale = nn.Sequential(
            BalanceConvBlock(256, 512),
            BalanceConvBlock(512, 256),
        )
        self._for_large_scale = nn.Sequential(
            BalanceConvBlock(256, 512),
            BalanceConvBlock(512, 256),
        )
        self._contract_medium = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            stride=2,
        )
        self._contract_large = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=2,
                stride=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
        )

        self._attention = ChannelScaleAttention()
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._x_reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        self._theta_reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        features_list = self._backbone(src)

        small_features = features_list[2] + self._for_small_scale(features_list[2])
        medium_features = features_list[1] + self._for_medium_scale(features_list[1])
        large_features = features_list[0] + self._for_large_scale(features_list[0])

        medium_features = self._contract_medium(medium_features)
        large_features = self._contract_large(large_features)

        combined_features = self._attention(
            [small_features, medium_features, large_features]
        )
        reduced_feature = self._avg_pool(combined_features).flatten(1)

        x_logits = self._x_reg_head(reduced_feature)
        theta_logits = self._theta_reg_head(reduced_feature)

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

    output = model(torch.rand(size=(2, 3, 1024, 1024)))

    print(output.shape)
