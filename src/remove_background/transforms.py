import math

import torch
from torchvision.transforms import v2
import torchvision.transforms as T
import numpy as np
import cv2

from ..utils.misc import resize_with_aspect_ratio


class ImageToArray(object):
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, img, tgt) -> tuple[np.ndarray, np.ndarray]:
        if self._normalize:
            img_tensor = np.array(img, dtype=np.float32) / 255.0
        else:
            img_tensor = np.array(img, dtype=np.uint8)

        tgt_tensor = np.array(tgt, dtype=np.float32)

        return img_tensor, tgt_tensor


class ColorJitter(object):
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self._adjust = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w, ch = img.shape
        img = self._adjust(img.reshape((ch, h, w))).reshape((h, w, ch))

        return img, tgt


class Rotate(object):
    def __init__(self, random_angle_range=(-20, 20)):
        self._random_angle_range = random_angle_range

    def __call__(self, img, tgt) -> tuple[np.ndarray, np.ndarray]:
        # angle in degrees
        angle = np.random.randint(*self._random_angle_range)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        # rotate the image with (intersection_x, height//2) as the center
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(img, rot_mat, (width, height))
        rotated_tgt = cv2.warpAffine(
            tgt, rot_mat, (width, height), flags=cv2.INTER_NEAREST
        )
        if img.ndim > rotated_img.ndim:
            rotated_img = rotated_img[..., None]

        return rotated_img, rotated_tgt


class RandomHorizontalFlip(object):
    def __init__(self, not_flip_prob=0.5):
        self._not_flip_prob = not_flip_prob

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self._not_flip_prob:
            img = np.flip(img, axis=1)
            tgt = np.flip(tgt, axis=1)

        return img, tgt


class PrintShape(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        print(img.shape, tgt.shape)
        return img, tgt


class RandomResizedCrop(object):
    def __init__(
        self,
        size: tuple[int] = (224, 224),
        scale: tuple[float] = (0.08, 1.0),
        ratio: tuple[float] = (0.75, 1.333),
        not_crop_prob: float = 0.2,
        number_of_classes: int = 20,
    ):
        self._size = size
        self._scale = scale
        self._ratio = ratio
        self._num_cls = number_of_classes
        self._attempt_limit = 50
        self._not_crop_prob = not_crop_prob

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width, _ = img.shape
        area = height * width

        resized_img, _ = resize_with_aspect_ratio(img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(tgt, target_size=self._size)
        resized_tgt = resized_tgt.clip(max=self._num_cls)
        if np.random.rand() < self._not_crop_prob:
            # do not crop
            if img.ndim > resized_img.ndim:
                resized_img = resized_img[..., None]
            return resized_img, resized_tgt

        for _ in range(self._attempt_limit):
            scale = np.random.uniform(self._scale[0], self._scale[1])
            ratio = np.random.uniform(self._ratio[0], self._ratio[1])

            new_area = area * scale
            new_height = int(round(math.sqrt(new_area * ratio)))
            new_width = int(round(math.sqrt(new_area / ratio)))
            if new_height >= height or new_width >= width:
                continue

            min_x = np.random.randint(0, width - new_width)
            min_y = np.random.randint(0, height - new_height)

            new_img = img[min_y : min_y + new_height, min_x : min_x + new_width, :]
            new_tgt = tgt[min_y : min_y + new_height, min_x : min_x + new_width]

            resized_img, _ = resize_with_aspect_ratio(new_img, target_size=self._size)
            resized_tgt, _ = resize_with_aspect_ratio(new_tgt, target_size=self._size)
            resized_tgt = resized_tgt.clip(max=self._num_cls)
            break

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt


class Resize(object):
    def __init__(self, size: tuple[int] = (256, 256), number_of_classes: int = 20):
        self._size = size
        self._num_cls = number_of_classes

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        resized_img, _ = resize_with_aspect_ratio(img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(tgt, target_size=self._size)
        resized_tgt = resized_tgt.clip(max=self._num_cls)

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt


class CenterCrop(object):
    def __init__(self, size: tuple[int] = (224, 224)):
        self._size = size

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = img.shape
        th, tw = self._size

        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))

        return img[i : i + th, j : j + tw], tgt[i : i + th, j : j + tw]


class ArrayToTensor(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img) / 255.0
        tgt = torch.from_numpy(tgt)

        # permute img to (C, H, W)
        return img.permute(2, 0, 1), tgt


class MaskToBinary(object):
    def __init__(self, *args, **kwargs):
        super(MaskToBinary, self).__init__(*args, **kwargs)

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # convert to binary mask
        # 0: background, 1: foreground
        tgt = (tgt > 0).astype(np.int64)

        return img, tgt


def build_transform():
    tr_fn = v2.Compose(
        [
            ImageToArray(normalize=False),
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(512, 512)),
            MaskToBinary(),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_valid_transform():
    tr_fn = v2.Compose(
        [
            Resize(size=(640, 640), number_of_classes=20),
            MaskToBinary(),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_scanned_transform():
    tr_fn = v2.Compose(
        [
            Rotate(),
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(640, 640), scale=(0.25, 1.0)),
            MaskToBinary(),
            ArrayToTensor(),
        ]
    )

    return tr_fn
