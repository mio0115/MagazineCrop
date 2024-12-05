import os

import cv2

from .arg_parser import get_parser
from .misc import combine_images


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

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

    for dir_name in dir_names:
        path_to_dir = os.path.join(os.getcwd(), "data", "processed_wo_resize", dir_name)
        path_to_save = os.path.join(os.getcwd(), "data", "combination", dir_name)
        if not os.path.isdir(path_to_dir):
            continue

        for image_name in os.listdir(path_to_dir):
            base_name = image_name.split(".")[0]
            color_yellow = (0, 255, 255)
            color_black = (0, 0, 0)

            original = cv2.imread(os.path.join(path_to_dir, base_name, "original.png"))
            left = cv2.imread(os.path.join(path_to_dir, base_name, "left.png"))
            right = cv2.imread(os.path.join(path_to_dir, base_name, "right.png"))

            pad_original = cv2.copyMakeBorder(
                original, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=color_yellow
            )
            pad_left = cv2.copyMakeBorder(
                left, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=color_yellow
            )
            pad_right = cv2.copyMakeBorder(
                right, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=color_yellow
            )

            output_image = combine_images(
                {"original": pad_original, "left": pad_left, "right": pad_right},
                padding_color=color_yellow,
                font_color=color_black,
            )

            cv2.imwrite(
                os.path.join(path_to_save, f"{base_name}.png"),
                output_image,
            )
