import os
import math

import cv2
from PIL import Image
import numpy as np

from .arg_parser import get_parser
from .misc import combine_images, combine_images_single_page


def scale_image(image, scale: float):
    height, width, _ = image.shape
    # compute the new area
    original_area = width * height
    resized_area = original_area // scale
    # compute the new shape
    new_height = int(math.sqrt(resized_area * height / width))
    new_width = int(resized_area / new_height)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _load_image(path: str) -> np.ndarray:
    try:
        pil_image = Image.open(path).convert("RGB")
        return np.array(pil_image, dtype=np.uint8)[..., ::-1]
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}. Error: {e}")


def _pad_image(
    image: np.ndarray,
    pad_size: int = 100,
    pad_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        pad_size,
        pad_size,
        pad_size,
        pad_size,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )


def combine_processed():
    args = get_parser("output").parse_args()

    if args.save_as is not None:
        path_to_save = os.path.join(args.output_dir, f"{args.save_as}.jpg")
    else:
        base_name = os.path.basename(args.original).split(".")[0]
        path_to_save = os.path.join(args.output_dir, f"{base_name}.jpg")
    # load images
    original: np.ndarray = _load_image(args.original)
    processed: list[np.ndarray] = [_load_image(path) for path in args.processed]

    # pad images for much clearer seperation
    pad_color = (0, 255, 255)  # yellow
    pad_original = _pad_image(original, pad_color=pad_color)
    pad_processed = [_pad_image(im, pad_color=pad_color) for im in processed]

    # combine original and processed images
    font_color = (0, 0, 0)  # black
    if len(pad_processed) == 2:
        output_image = combine_images(
            {
                "original": pad_original,
                "left": pad_processed[0],
                "right": pad_processed[1],
            },
            padding_color=pad_color,
            font_color=font_color,
        )
    elif len(pad_processed) == 1:
        output_image = combine_images_single_page(
            images={"original": pad_original, "page": pad_processed[0]},
            padding_color=pad_color,
            font_color=font_color,
        )
    else:
        raise ValueError("The number of processed images should be 1 or 2.")

    resized_output = scale_image(output_image, args.scale)
    Image.fromarray(resized_output[..., ::-1]).save(
        path_to_save,
        format="JPEG",
        quality=80,
        subsampling=2,
        optimize=True,
        progressive=True,
    )
    print(f"Saved to {path_to_save}")


if __name__ == "__main__":
    combine_processed()

    # dir_names = [
    #     "B6855",
    #     "B6960",
    #     "B6995",
    #     "C2789",
    #     "C2797",
    #     "C2811",
    #     "C2926",
    #     "C3909",
    #     "C3920",
    # ]
    # addtional_dir_names = ["free_talk_03", "free_talk_04"]

    # single_pages = [
    #     "no7-1009_105331",
    #     "no7-1009_105616",
    #     "no7-1008_135617",
    #     "no7-1008_143236",
    #     "no7-1007_151540",
    #     "no7-1007_153209",
    #     "no6-1009_173556",
    #     "no6-1009_173948",
    #     "no6-1011_092006",
    #     "no6-1011_092220",
    #     "no6-1011_095239",
    #     "no6-1011_100319",
    #     "no7-1004_165405",
    # ]

    # for dir_name in addtional_dir_names:
    #     path_to_dir = os.path.join(os.getcwd(), "data", "processed_wo_resize", dir_name)
    #     path_to_save = os.path.join(os.getcwd(), "data", "combination", f"{dir_name}")
