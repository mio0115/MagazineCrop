import os
import math

import torch
from torch import nn
import cv2
import numpy as np

from .model.model import build_model
from ..utils.arg_parser import get_parser

# to download model's weights, execute the following command:
# scp <username>@<ip>:/home/ubuntu/projects/MagazineCrop/src/remove_background/checkpoints/<model_name> ./src/remove_background/checkpoints/


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    new_width, new_height = 512, 512

    cv2.namedWindow("splitted image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("splitted image", width=new_width, height=new_height)

    # path_to_images = os.path.join(os.getcwd(), "data", "example")

    path_to_image = os.path.join(
        os.getcwd(),
        "data",
        "train_data",
        "scanned",
        "images",
        "B6960",
        "no7-1008_135655.tif",
    )
    image = cv2.imread(os.path.join(path_to_image))

    height, width, *_ = image.shape

    # image = cv2.imread(os.path.join(path_to_images, args.image_name))

    resized_image = cv2.resize(image, (new_width, new_height))

    model = build_model()
    model.load_state_dict(
        torch.load(
            os.path.join(args.path_to_model_dir, args.model_name), weights_only=True
        )
    )
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        in_image = (
            torch.tensor(resized_image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        )
        in_image = in_image.to(args.device)
        logits = model(in_image)
        line_coords = logits.sigmoid().cpu().numpy()

        line_x_coord = int(line_coords[..., 0] * width)
        line_theta = line_coords[..., 1] * math.pi

        point_0 = (line_x_coord, height // 2)

        line_slope = np.tan(line_theta)

        point_1 = (line_x_coord - 10, int(height // 2 - line_slope * 10))
        point_2 = (line_x_coord + 10, int(height // 2 + line_slope * 10))

        cv2.line(image, pt1=point_1, pt2=point_2, color=(0, 0, 255), thickness=3)

        # cv2.imshow("masked image", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("splitted image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
