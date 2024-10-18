import math

import torch
from torchvision.transforms import v2
import torchvision.transforms as T
import numpy as np
import cv2


class ImageToArray(object):
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, img, tgt) -> tuple[np.ndarray, np.ndarray]:
        if self._normalize:
            img_tensor = np.array(img, dtype=np.float32) / 255.0
        else:
            img_tensor = np.array(img, dtype=np.uint8)

        tgt_tensor = np.array(tgt, dtype=np.int64)

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


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self._prob = probability

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self._prob:
            img = np.flip(img, axis=1)
            tgt = np.flip(tgt, axis=1)

        return img, tgt


class RandomResizedCrop(object):
    def __init__(
        self,
        size: tuple[int] = (224, 224),
        scale: tuple[float] = (0.08, 1.0),
        ratio: tuple[float] = (0.75, 1.333),
        number_of_classes: int = 20,
    ):
        self._size = size
        self._scale = scale
        self._ratio = ratio
        self._num_cls = number_of_classes

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width, _ = img.shape
        scale = np.random.uniform(self._scale[0], self._scale[1])
        ratio = np.random.uniform(self._ratio[0], self._ratio[1])
        new_height = min(int(height * scale * math.sqrt(ratio)), height)
        new_width = min(int(width * scale / math.sqrt(ratio)), width)

        min_x = np.random.randint(0, width - new_width + 1)
        min_y = np.random.randint(0, height - new_height + 1)

        new_img = img[min_y : min_y + new_height, min_x : min_x + new_width, :]
        new_tgt = tgt[min_y : min_y + new_height, min_x : min_x + new_width]

        resized_img = cv2.resize(new_img, self._size, interpolation=cv2.INTER_LINEAR)
        resized_tgt = cv2.resize(
            new_tgt, self._size, interpolation=cv2.INTER_NEAREST
        ).clip(max=self._num_cls)

        return resized_img, resized_tgt


class Resize(object):
    def __init__(self, size: tuple[int] = (256, 256), number_of_classes: int = 20):
        self._size = size
        self._num_cls = number_of_classes

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        resized_img = cv2.resize(img, self._size, interpolation=cv2.INTER_LINEAR)
        resized_tgt = cv2.resize(tgt, self._size, interpolation=cv2.INTER_NEAREST).clip(
            max=self._num_cls
        )

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

        return img, tgt


class MaskToBinary(object):
    def __init__(self, number_of_classes: int, *args, **kwargs):
        super(MaskToBinary, self).__init__(*args, **kwargs)

        self._num_cls = number_of_classes

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # convert to binary mask
        # 0: background, 1: foreground
        tgt = (tgt > 0).astype(np.int64)

        return img, tgt


def build_transforms(args):
    tr_fn = v2.Compose(
        [
            ImageToArray(normalize=False),
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(512, 512)),
            MaskToBinary(number_of_classes=20),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_valid_transform(args):
    tr_fn = v2.Compose(
        [
            ImageToArray(normalize=False),
            Resize(size=(512, 512), number_of_classes=20),
            CenterCrop(size=(496, 496)),
            MaskToBinary(number_of_classes=20),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_scanned_transforms():
    tr_fn = v2.Compose(
        [
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(512, 512), scale=(0.25, 1.0)),
            ArrayToTensor(),
        ]
    )

    return tr_fn
