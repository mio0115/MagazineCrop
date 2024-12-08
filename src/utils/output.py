import os
import math

import cv2
from PIL import Image
import numpy as np

from .arg_parser import get_parser
from .misc import combine_images, combine_images_single_page


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

    single_pages = [
        "no7-1009_105331",
        "no7-1009_105616",
        "no7-1008_135617",
        "no7-1008_143236",
        "no7-1007_151540",
        "no7-1007_153209",
        "no6-1009_173556",
        "no6-1009_173948",
        "no6-1011_092006",
        "no6-1011_092220",
        "no6-1011_095239",
        "no6-1011_100319",
        "no7-1004_165405",
    ]

    for dir_name in dir_names:
        path_to_dir = os.path.join(os.getcwd(), "data", "processed_wo_resize", dir_name)
        path_to_save = os.path.join(os.getcwd(), "data", "combination", f"{dir_name}")
        if not os.path.isdir(path_to_dir):
            continue

        for image_name in os.listdir(path_to_dir):
            if args.num_pages == 1 and image_name not in single_pages:
                continue
            if args.num_pages == 2 and image_name in single_pages:
                continue

            base_name = image_name.split(".")[0]
            color_yellow = (0, 255, 255)
            color_black = (0, 0, 0)

            # original = cv2.imread(os.path.join(path_to_dir, base_name, "original.png"))
            # left = cv2.imread(os.path.join(path_to_dir, base_name, "left.png"))
            # right = cv2.imread(os.path.join(path_to_dir, base_name, "right.png"))
            if args.num_pages == 2:
                pil_original = Image.open(
                    os.path.join(path_to_dir, base_name, "original.tif")
                )
                pil_left = Image.open(os.path.join(path_to_dir, base_name, "left.tif"))
                pil_right = Image.open(
                    os.path.join(path_to_dir, base_name, "right.tif")
                )

                original = np.array(pil_original, dtype=np.uint8)[..., ::-1]
                left = np.array(pil_left, dtype=np.uint8)[..., ::-1]
                right = np.array(pil_right, dtype=np.uint8)[..., ::-1]

                pad_original = cv2.copyMakeBorder(
                    original,
                    100,
                    100,
                    100,
                    100,
                    cv2.BORDER_CONSTANT,
                    value=color_yellow,
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
            else:
                pil_original = Image.open(
                    os.path.join(path_to_dir, base_name, "original.tif")
                )
                pil_page = Image.open(os.path.join(path_to_dir, base_name, "page.tif"))
                original = np.array(pil_original, dtype=np.uint8)[..., ::-1]
                page = np.array(pil_page, dtype=np.uint8)[..., ::-1]

                pad_original = cv2.copyMakeBorder(
                    original,
                    100,
                    100,
                    100,
                    100,
                    cv2.BORDER_CONSTANT,
                    value=color_yellow,
                )
                pad_page = cv2.copyMakeBorder(
                    page, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=color_yellow
                )

                output_image = combine_images_single_page(
                    images={"original": pad_original, "page": pad_page},
                    padding_color=color_yellow,
                    font_color=color_black,
                )

            height, width, _ = output_image.shape
            area_scale = 15
            original_area = width * height
            resized_area = original_area // area_scale
            new_height = int(math.sqrt(resized_area * height / width))
            new_width = int(resized_area / new_height)

            resized_output = cv2.resize(
                output_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            # print(f"{width=}, {height=}")
            # print(f"{new_width=}, {new_height=}")

            # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("output", 1024, 1024)
            # cv2.imshow("output", resized_output)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print(f"save to {os.path.join(path_to_save, f"{base_name}.jpg")}")

            Image.fromarray(resized_output[..., ::-1]).save(
                os.path.join(path_to_save, f"{base_name}.jpg"),
                format="JPEG",
                quality=80,
                subsampling=2,
                optimize=True,
                progressive=True,
            )
