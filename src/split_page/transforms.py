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
    ):
        self._size = size
        self._scale = scale
        self._ratio = ratio

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

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt


class ArrayToTensor(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img) / 255.0
        tgt = torch.from_numpy(tgt)

        return img, tgt


def build_scanned_transforms():
    tr_fn = v2.Compose(
        [
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
            ArrayToTensor(),
        ]
    )

    return tr_fn
