import os

import torch
from torch import nn
import cv2
import numpy as np

from .model.model_unet_pp import build_model
from ..utils.arg_parser import get_parser
from ..utils.misc import resize_with_aspect_ratio

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 1024, 1024
    dir_names = [
        "B6855",
        "B6960",
        "B6995",
        "C2789",
        "C2797",
        "C2811",
        "C2926",
        "C3909",
        "C3920",
    ]
    path_to_model = os.path.join(
        os.getcwd(), "src", "remove_background", "checkpoints", args.model_name
    )

    model = torch.load(path_to_model, weights_only=False)
    model.to(args.device)
    model.eval()

    for dir_name in dir_names:
        path_to_dir = os.path.join(
            os.getcwd(), "data", "valid_data", "scanned", "images", dir_name
        )
        if not os.path.isdir(path_to_dir):
            continue

        for image_name in os.listdir(path_to_dir):
            if not image_name.endswith(".tif"):
                continue

            image = cv2.imread(
                os.path.join(path_to_dir, image_name), cv2.IMREAD_GRAYSCALE
            )
            resized_image, _ = resize_with_aspect_ratio(
                image, target_size=(new_width, new_height)
            )

            eh_image = cv2.equalizeHist(resized_image)

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 2048, 1024)

            with torch.no_grad():
                in_image = torch.tensor(eh_image)[None, :, :, None].float() / 255.0
                in_image = in_image.to(args.device)
                logits = model(in_image)
                is_fg_prob = logits.sigmoid().squeeze().cpu().numpy()
            fg_mask = is_fg_prob >= 0.4

            masked_image = resized_image.copy()
            masked_image[~fg_mask] = 0

            mask = fg_mask.astype(np.uint8) * 255

            cv2.imshow("image", cv2.hconcat([eh_image, mask, masked_image]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
