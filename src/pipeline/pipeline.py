import os
from functools import partial

from PIL import Image
import numpy as np
import cv2

from ..remove_background.infer import PredictForeground
from ..split_page.infer import SplitPage
from ..dewarp_curl.dewarp import Dewarp
from .combination import Combination
from ..utils.arg_parser import get_parser
from ..utils.misc import compute_resized_shape
from ..utils.save_output import save_mask


def load_image(path: str) -> np.ndarray:
    """
    Load and convert an image from a given path into a NumPy array with BGR order.
    """
    pil_image = Image.open(path).convert("RGB")
    # convert the image to numpy array and change the channel order from RGB to BGR
    image = np.array(pil_image, dtype=np.uint8)[..., ::-1]

    return image


def get_arguments(parser_mode: str = "user"):
    """
    Get the arguments from the argument parser.
    """
    parser = get_parser(parser_mode)
    return parser.parse_args()


class MagazineCropPipeline:
    """
    Orchestrates the magazine cropping pipeline:
    1. Parse arguments.
    2. Validate input/output paths.
    3. Initialize required models and classes.
    4. Process the image (split, fix distortion).
    5. Save the resulting pages.
    """

    def __init__(
        self, parser_mode: str = "user", new_shape: tuple[int, int] = (1024, 1024)
    ):
        self._tmp_shape = new_shape

        args = get_arguments(parser_mode)
        self.validate_arguments(args)
        self.set_instance_variables(args)

        # Instance variables of pipeline components
        self._split_pages = None
        self._fix_distortion = None

    def validate_arguments(self, args):
        """
        Validate the arguments from argparse to ensure input/output paths are valid.
        """
        if not os.path.isdir(args.output_dir):
            raise FileNotFoundError(f"{args.output_dir} does not exist")

        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"{args.input} does not exist")

        if not isinstance(args.output_scale, (int, float)) or (
            isinstance(args.output_scale, (int, float))
            and not (0 < args.output_scale <= 1)
        ):
            raise ValueError("Output scale must be a float between 0 and 1")

        if not isinstance(args.quality, int) or (
            isinstance(args.quality, int) and not (0 < args.quality <= 100)
        ):
            raise ValueError("Quality must be an integer between 0 and 100")

    def set_instance_variables(self, args):
        """
        Set the instance variables based on the given arguments.
        """
        self._input_path = args.input
        self._output_dir = args.output_dir
        self._output_scale = args.output_scale
        self._quality = args.quality
        self._verbose = args.verbose

        self._device = "cuda" if args.use_gpu else "cpu"
        self._num_pages = 1 if args.single_page else 2
        self._image_name = os.path.basename(self._input_path).split(".")[0]
        self._save_mask_fn = partial(save_mask, path_to_save=self._output_dir)

    def init_pipeline(self):
        """
        Initialize the pipeline components based on the given arguments.
        """
        predict_fg = PredictForeground(
            device=self._device, verbose=self._verbose, new_size=self._tmp_shape
        )
        predict_sp = SplitPage(
            device=self._device, verbose=self._verbose, new_size=self._tmp_shape
        )
        self._split_pages = Combination(
            predict_fg,
            predict_sp,
            num_pages=self._num_pages,
            verbose=self._verbose,
            save_mask_fn=self._save_mask_fn,
        )
        self._fix_distortion = Dewarp(target_size=self._tmp_shape)

    def process(self, image: np.ndarray):
        """
        Process the input image in the following steps:
        1. Remove background and split the image into pages if needed.
        2. Fix the distortion of the pages.
        """
        # Remove background and split the image into pages if needed
        pages = self._split_pages(image, is_gray=False)
        # Fix the distortion of the pages
        fixed_pages = []
        for page in pages:
            fixed_pages.append(self._fix_distortion(page["image"], page["mask"]))

        return fixed_pages

    def save_output(
        self,
        fixed_pages: list[np.ndarray],
    ):
        """
        Scaled and save the result images to the output directory.
        """
        # Scale output if needed
        scaled_pages = []
        for page in fixed_pages:
            if self._output_scale == 1.0:
                scaled_pages.append(page)
                continue

            (new_height, new_width) = compute_resized_shape(
                shape=page.shape, scale=self._output_scale
            )
            scaled_pages.append(cv2.resize(page, (new_width, new_height)))

        # Save the output images
        for i, page in enumerate(scaled_pages):
            Image.fromarray(page[..., ::-1]).save(
                os.path.join(self._output_dir, f"{self._image_name}_{i}.tif"),
                format="TIFF",
                compression="jpeg",
                quality=self._quality,
            )

    def __call__(self):
        """
        Execute the entire pipeline:
        1. Initialize pipeline components
        2. Load input image
        3. Process image
        4. Save results
        """
        # Initialize the pipeline components
        self.init_pipeline()

        # Load the input image from the given path
        image = load_image(path=self._input_path)
        # Process the image
        fixed_pages = self.process(image=image)
        # Scaled (if needed) and save the result images
        self.save_output(fixed_pages)
