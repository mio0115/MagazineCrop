import os
import time
from functools import partial

import cv2
import numpy as np
from PIL import Image

from .remove_background.infer import PredictForeground
from .split_page.infer import SplitPage
from .fix_distortion.fix_distortion import FixDistortion
from .combination import Combination
from .utils.arg_parser import get_parser
from .utils.misc import compute_resized_shape
from .utils.save_output import save_mask


def main():
    start_time = time.time()
    parser = get_parser("user")
    args = parser.parse_args()

    # check if the input image exists
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"{args.output_dir} does not exist")
    # check if the output directory exists
    try:
        pil_image = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"{args.input} does not exist")
    except Exception as e:
        raise e

    # initialize the pipeline
    new_shape = (1024, 1024)  # width, height
    device = "cuda" if args.use_gpu else "cpu"
    predict_fg = PredictForeground(
        device=device, verbose=args.verbose, new_size=new_shape
    )
    predict_sp = SplitPage(device=device, verbose=args.verbose, new_size=new_shape)
    split_pages = Combination(
        predict_fg,
        predict_sp,
        num_pages=1 if args.single_page else 2,
        verbose=args.verbose,
        save_mask_fn=partial(save_mask, path_to_save=args.output_dir),
    )
    fix_distortion = FixDistortion(target_size=new_shape)

    # convert the image to numpy array and change the channel order
    image = np.array(pil_image, dtype=np.uint8)[..., ::-1]
    # split the image into pages and fix the distortion
    try:
        pages = split_pages(image, is_gray=False)

        fixed_pages = []
        for page in pages:
            fixed_pages.append(fix_distortion(page["image"], page["mask"]))
    except Exception as e:
        raise e

    # scale the output images
    scaled_pages = []
    for page in fixed_pages:
        (new_height, new_width) = compute_resized_shape(
            shape=page.shape, scale=args.output_scale
        )
        scaled_pages.append(
            cv2.resize(page, (new_width, new_height), interpolation=cv2.INTER_AREA)
        )

    # save the output images
    image_name = os.path.basename(args.input).split(".")[0]
    for i, page in enumerate(scaled_pages):
        Image.fromarray(page[..., ::-1]).save(
            os.path.join(args.output_dir, f"{image_name}_{i}.tif"),
            format="TIFF",
            compression="jpeg",
            quality=args.quality,
        )

    end_time = time.time()
    print(f"Completed in {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
