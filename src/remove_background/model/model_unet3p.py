from ...model_templates.templates import UNet3Plus


def build_model(in_channels: int = 3, out_channels: int = 1):
    model = UNet3Plus(in_channels=in_channels, out_channels=out_channels)

    return model
