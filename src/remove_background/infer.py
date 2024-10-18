import os

import torch
from torch import nn
import cv2

from .model.model_unet_pp import build_unetplusplus
from ..utils.arg_parser import get_parser

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_images = os.path.join(os.getcwd(), "data", "example")

    image = cv2.imread(os.path.join(path_to_images, args.image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (1024, 1024))
    cv2.imshow("original image", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

    model = build_unetplusplus(number_of_classes=1)
    model.load_state_dict(
        torch.load(
            os.path.join(args.path_to_model_dir, args.model_name), weights_only=True
        )
    )
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        in_image = torch.tensor(resized_image).unsqueeze(0).float()
        in_image = in_image.to(args.device)
        mask = model(in_image)
        mask = mask.sigmoid().squeeze().cpu().numpy()
        mask = mask >= 0.5

        masked_image = resized_image.copy()
        masked_image[~mask] = 0

        cv2.imshow("masked image", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
