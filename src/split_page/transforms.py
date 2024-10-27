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

        tgt_tensor = np.array(tgt, dtype=np.float32)

        return img_tensor, tgt_tensor


class Rotate(object):
    def __init__(self, random_angle_range=(-20, 20)):
        self._random_angle_range = random_angle_range

    def __call__(self, img, tgt) -> tuple[np.ndarray, np.ndarray]:
        # angle in degrees
        angle = np.random.randint(*self._random_angle_range)
        height, width = img.shape[:2]
        center = (int(tgt[..., 0]), height // 2)
        # rotate the image with (intersection_x, height//2) as the center
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(img, rot_mat, (width, height))

        arc_angle = np.deg2rad(angle)
        rotated_tgt = tgt.copy()
        rotated_tgt[..., 1] = tgt[..., 1] - arc_angle

        return rotated_img, rotated_tgt


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self._prob = probability

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self._prob:
            img = np.flip(img, axis=1)

            tgt[..., 0] = img.shape[1] - tgt[..., 0]
            tgt[..., 1] = np.pi - tgt[..., 1]

        return img, tgt


class PrintShape(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        print(img.shape, tgt.shape)
        return img, tgt


class PrintTarget(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        print(tgt)
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
        area = height * width

        new_area = area * scale
        new_height = int(round(math.sqrt(new_area * ratio)))
        new_width = int(round(math.sqrt(new_area / ratio)))
        if new_height > height or new_width > width:
            new_height = min(new_height, height)
            new_width = min(new_width, width)

        tgt_x, tgt_y = int(tgt[..., 0]), int(height / 2)
        min_x_low = max(tgt_x - int(0.2 * new_width), 0)
        min_x_high = min(int(0.9 * tgt_x), width - new_width + 1)
        if min_x_low >= min_x_high:
            min_x_low, min_x_high = 0, width - new_width + 1

        min_y_low = max(tgt_y - int(0.2 * new_height), 0)
        min_y_high = min(int(0.9 * tgt_y), height - new_height + 1)
        if min_y_low >= min_y_high:
            min_y_low, min_y_high = 0, height - new_height + 1

        min_x = np.random.randint(min_x_low, min_x_high)
        min_y = np.random.randint(min_y_low, min_y_high)

        new_img = img[min_y : min_y + new_height, min_x : min_x + new_width, :]

        resized_img = cv2.resize(new_img, self._size, interpolation=cv2.INTER_LINEAR)
        resized_tgt = tgt.copy()
        resized_tgt[..., 0] = (resized_tgt[..., 0] - min_x) * self._size[1] / new_width

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt


class ArrayToTensor(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img) / 255.0
        tgt = torch.from_numpy(tgt)
        tgt[..., 0] /= img.shape[1]

        img = img.permute(2, 0, 1).float().contiguous()

        return img, tgt


def build_scanned_transforms():
    tr_fn = v2.Compose(
        [
            ImageToArray(normalize=False),
            Rotate(),
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(512, 512), scale=(0.3, 1.0)),
            ArrayToTensor(),
        ]
    )

    return tr_fn
