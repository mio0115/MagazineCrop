import os

import cv2
from PIL import Image


def save_mask(
    image,
    mask,
    path_to_save,
    name: str,
    pad_color: tuple[int, int, int] = (0, 255, 255),
):
    masked_image = (image * mask[..., None]).astype("uint8")

    output = cv2.hconcat(
        [
            cv2.copyMakeBorder(
                image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=pad_color
            ),
            cv2.copyMakeBorder(
                masked_image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=pad_color
            ),
        ]
    )

    Image.fromarray(output[..., ::-1]).save(
        os.path.join(path_to_save, name), format="TIFF", compression="jpeg", quality=70
    )


def save_line(*args, **kwargs):
    pass
