import torch
from torch import nn
from torchvision import models as tv_models

from .blocks.blocks import (
    DoubleConvBlock,
    ContractBlock,
    BalanceConvBlock,
    SpatialScaleAttention,
    ConvBlock,
)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, *args, **kwargs):
        super(UNet3Plus, self).__init__(*args, **kwargs)

        if in_channels != 3:
            self._warmup = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self._warmup = nn.Identity()
        self._downsample, self._upsample = self._init_sampling_layers()
        self.encoders, self.out_channels = self._init_encoders()
        self.decoders = self._init_decoders(num_layers=4, channels=320)
        self.to_logits = self._init_logits(out_channels)
        self.connector_layers = self._init_connectors()

    def _init_sampling_layers(self):
        """Initializes downsample and upsample layers."""
        scales = [1, 2, 4, 8, 16]
        downsample = nn.ModuleDict(
            {
                f"{i}": (
                    nn.MaxPool2d(kernel_size=s, stride=s) if i >= 1 else nn.Identity()
                )
                for i, s in enumerate(scales)
            }
        )
        upsample = nn.ModuleDict(
            {
                f"{i}": (
                    nn.Upsample(scale_factor=s, mode="bilinear")
                    if i >= 1
                    else nn.Identity()
                )
                for i, s in enumerate(scales)
            }
        )
        return downsample, upsample

    def _init_encoders(self):
        """Initializes VGG16 backbone as the encoder."""
        vgg16 = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        encoders = nn.ModuleList(
            [vgg16[:4], vgg16[5:9], vgg16[10:16], vgg16[17:23], vgg16[24:30]]
        )
        return encoders, [64, 128, 256, 512, 512]

    def _init_decoders(self, num_layers, channels):
        """Creates decoder blocks which is aggregation layers followed by decoders."""
        return nn.ModuleList(
            [
                DoubleConvBlock(
                    in_channels=channels,
                    inter_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    normalization="group",
                    activation_fn=nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

    def _init_logits(self, out_channels):
        """Initializes final logits layers."""
        logits = {
            f"{i}": nn.Sequential(
                self._upsample[f"{i-1}"],
                nn.Conv2d(320, out_channels, kernel_size=3, padding=1),
            )
            for i in range(1, 5)
        }
        logits["5"] = nn.Sequential(
            self._upsample["4"], nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        )
        return nn.ModuleDict(logits)

    def _init_connectors(self):
        """Creates connector layers for full-scale skip connections."""
        connectors = {}
        for tgt in range(1, 5):  # Decoder levels
            for src in range(1, 6):  # Encoder levels (including final feature map)
                key = f"d{tgt}e{src}" if src <= tgt or src == 5 else f"d{tgt}d{src}"
                connectors[key] = self._create_connector(src, tgt)
        return nn.ModuleDict(connectors)

    def _create_connector(self, src, tgt):
        """Creates a connector layer for encoder-decoder fusion."""
        scale_layer = (
            self._downsample[f"{abs(src - tgt)}"]
            if src < tgt
            else self._upsample[f"{abs(src - tgt)}"]
        )
        conv_layer = nn.Conv2d(
            self.out_channels[src - 1] if src <= tgt or src == 5 else 320,
            64,
            kernel_size=3,
            padding=1,
        )
        return nn.Sequential(scale_layer, conv_layer)

    def forward(self, src: torch.Tensor):
        """Forward pass through UNet3+."""
        enc_outputs, dec_outputs = [], {}

        x = self._warmup(src)

        # Encoder pass
        for encoder in self.encoders:
            x = encoder(x)
            enc_outputs.append(x)
            x = self._downsample["1"](x)

        logits = [self.to_logits["5"](enc_outputs[4])]

        # Decoder pass
        for d in range(4, 0, -1):
            feats = [
                self.connector_layers[f"d{d}e{s}"](enc_outputs[s - 1])
                for s in range(1, d + 1)
            ]
            for layer in range(d + 1, 5):
                feats.append(
                    self.connector_layers[f"d{d}d{layer}"](dec_outputs[f"{layer}"])
                )
            feats.append(self.connector_layers[f"d{d}e5"](enc_outputs[4]))

            combined_feats = self.decoders[d - 1](torch.cat(feats, 1))
            logits.append(self.to_logits[f"{d}"](combined_feats))
            dec_outputs[f"{d}"] = combined_feats

        outputs = {"logits": logits}
        return outputs


class UNetPlusPlus(nn.Module):
    def __init__(
        self, contract_block: nn.Module, embed_dims: list[int], *args, **kwargs
    ):
        # embed_dims: [32, 64, 128, 256, 512]
        super(UNetPlusPlus, self).__init__(*args, **kwargs)

        self._contract_blk = contract_block

        self._expand_blks = nn.ModuleList()
        self._pyramid_height = len(embed_dims) - 1

        for level_idx in range(self._pyramid_height):
            curr_lvl_ch = embed_dims[level_idx]
            next_lvl_ch = embed_dims[level_idx + 1]
            expand_level = nn.ModuleList()

            num_nested_convs = self._pyramid_height - level_idx
            for nested_idx in range(num_nested_convs):
                factor = nested_idx + 1

                expand_block = nn.ModuleDict(
                    {
                        "scale_attention": (
                            SpatialScaleAttention(
                                in_channels=curr_lvl_ch, factor=factor
                            )
                            if factor > 1
                            else nn.Identity()
                        ),
                        "conv_block": DoubleConvBlock(
                            in_channels=curr_lvl_ch + next_lvl_ch,
                            inter_channels=curr_lvl_ch,
                            out_channels=curr_lvl_ch,
                            bias=False,
                            activation_fn=[nn.ReLU(), None],
                            normalization="batch",
                            # normalization="group",
                        ),
                    }
                )
                expand_level.append(expand_block)
            self._expand_blks.append(expand_level)

        self._weighted_sum = SpatialScaleAttention(
            in_channels=embed_dims[0], factor=self._pyramid_height
        )
        self._weights = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.out_channels = embed_dims[0]

        self._upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, src):
        # execute contractive block
        src_list: list[torch.Tensor] = self._contract_blk(src)
        nested_features: list[list[torch.Tensor]] = [[feature] for feature in src_list]

        # execute expansive block
        for pyramid_lvl in range(self._pyramid_height - 1, -1, -1):
            for blk_ind in range(self._pyramid_height - pyramid_lvl):
                features = nested_features[pyramid_lvl]
                # apply the scale attention to the current feature map to reduce the channels
                x = self._expand_blks[pyramid_lvl][blk_ind]["scale_attention"](features)
                # concatenate the upsampled feature map from the previous level and the current feature map
                upsampled = self._upsample(nested_features[pyramid_lvl + 1][blk_ind])
                concat_feature = torch.cat(
                    (x if isinstance(x, list) else [x]) + [upsampled], 1
                )
                # apply the double convolution block to the concatenated feature map
                x = self._expand_blks[pyramid_lvl][blk_ind]["conv_block"](
                    concat_feature
                )

                nested_features[pyramid_lvl].append(x)

        # apply the weighted sum to the output feature maps
        output_logits = self._weighted_sum(nested_features[0][1:])
        return output_logits


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
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(2048, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                ),
            ]
        )
        self._conv3x3 = nn.ModuleList(
            [
                BalanceConvBlock(512, 256),
                BalanceConvBlock(512, 256),
                BalanceConvBlock(512, 256),
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
    in_channels: int = 3,
    embed_dims: list[int] = [32, 64, 128, 256, 512],
) -> UNetPlusPlus:
    contract_block = ContractBlock(input_dim=in_channels, embed_dims=embed_dims)
    model = UNetPlusPlus(contract_block, embed_dims)

    return model


def build_fpn() -> FeaturePyramidNetwork:
    model = FeaturePyramidNetwork()

    return model


if __name__ == "__main__":
    # model = FeaturePyramidNetwork()
    model = UNet3Plus(out_channels=1)

    model = model.to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for e in range(20):
            t = torch.rand(2, 3, 480, 480).to("cuda")
            out = model(t)

            for logit in out:
                print(logit.shape)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")

    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {trainable_params:,}")

    # print(out[0].shape, out[1].shape, out[2].shape)

    # model = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
    # print(model.features[0])
    # for name, module in model.named_children():
    #     print(f"{name}: {module}")
