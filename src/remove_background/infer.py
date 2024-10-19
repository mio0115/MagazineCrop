import os

import torch
from torch import nn
import cv2
import numpy as np

from .model.model_unet_pp import build_unetplusplus
from ..utils.arg_parser import get_parser

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_images = os.path.join(os.getcwd(), "data", "example")

    image = cv2.imread(os.path.join(path_to_images, args.image_name))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (2048, 2048))
    _, thres_image = cv2.threshold(
        cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU
    )
    # cv2.imshow("original image", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("original image", resized_image)
    cv2.imshow("thresholded image", thres_image)

    model = build_unetplusplus(number_of_classes=1)
    model.load_state_dict(
        torch.load(
            os.path.join(args.path_to_model_dir, args.model_name), weights_only=True
        )
    )
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        in_image = (
            torch.tensor(cv2.cvtColor(thres_image, cv2.COLOR_GRAY2BGR))
            .unsqueeze(0)
            .float()
            / 255.0
        )
        in_image = in_image.to(args.device)
        logits = model(in_image)
        is_fg_prob = logits.sigmoid().squeeze().cpu().numpy()
        fg_mask = is_fg_prob >= 0.5

        print(np.unique(fg_mask))

        masked_image = resized_image.copy()
        masked_image[~fg_mask] = 0

        # cv2.imshow("masked image", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("masked image", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
