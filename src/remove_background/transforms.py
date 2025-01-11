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

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        rotated_weights = cv2.warpAffine(
            weights, rot_mat, (width, height), flags=cv2.INTER_AREA
        )
        if img.ndim > rotated_img.ndim:
            rotated_img = rotated_img[..., None]

        return rotated_img, rotated_tgt, rotated_weights


class RandomHorizontalFlip(object):
    def __init__(self, not_flip_prob=0.5):
        self._not_flip_prob = not_flip_prob

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self._not_flip_prob:
            img = np.flip(img, axis=1)
            tgt = np.flip(tgt, axis=1)
            weights = np.flip(weights, axis=1)

        return img, tgt, weights


class PrintShape(object):
    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        print(img.shape, tgt.shape)
        return img, tgt


def crop_arrays(coords: tuple[int], *arrays):
    # coords: (min_y, min_x, height, width)
    top, left, height, width = coords
    cropped_arrays = []

    for array in arrays:
        cropped_arrays.append(
            array[
                top : top + height,
                left : left + width,
                ...,
            ]
        )

    return cropped_arrays


def mask_padded_region(arr: np.ndarray, pad_coords: tuple[int]):
    """
    Zero out the region outside the actual image after padding.
    pad_coords: (top, bottom, left, right)
    """
    top, bottom, left, right = pad_coords

    mask = np.zeros_like(arr, dtype=np.int32)
    mask[top:bottom, left:right] = 1

    return arr * mask


class RandomResizedCrop(object):
    """
    A transformation that randomly decides among:
      1) Not cropping at all.
      2) Cropping based on 'margin' logic.
      3) Cropping based on 'bookmark' logic (if available).
      4) Cropping via random scale & ratio.

    Then it resizes the final crop to `self._size`.

    Args:
        size (tuple[int]):        Final size (height, width) after resizing.
        scale (tuple[float]):     Range of area scale for random scale&ratio crop.
        ratio (tuple[float]):     Range of aspect ratio for random scale&ratio crop.
        not_crop_prob (float):    Probability of not cropping at all.
        crop_margin_prob (float): Probability of doing a margin-based crop.
        crop_bookmark_prob (float): Probability of doing a bookmark-based crop.
        bookmark_label (int):     Class label for 'bookmark'; must be != -1 if using bookmark cropping.
        number_of_classes (int):  Used to clamp target label after resizing.
    """

    def __init__(
        self,
        size: tuple[int] = (224, 224),
        scale: tuple[float] = (0.08, 1.0),
        ratio: tuple[float] = (0.75, 1.333),
        not_crop_prob: float = 0.2,
        crop_margin_prob: float = 0.3,
        crop_bookmark_prob: float = -1,
        bookmark_label: int = -1,
        number_of_classes: int = 20,
    ):
        self._size = size
        self._scale = scale
        self._ratio = ratio
        # self._num_cls = number_of_classes
        self._attempt_limit = 100

        self._not_crop_prob = not_crop_prob
        self._crop_margin_prob = crop_margin_prob
        self._crop_bookmark_prob = crop_bookmark_prob
        self._bookmark_label = bookmark_label

        if self._bookmark_label == -1 and self._crop_bookmark_prob > 0:
            raise ValueError("bookmark_label must be set to a valid class label")

    def _no_crop(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simply resize the entire image, target, and weights to `self._size`.
        """
        resized_img, _ = resize_with_aspect_ratio(img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(
            tgt, target_size=self._size, interpolation=cv2.INTER_NEAREST
        )
        resized_weights, _, pad_coords = resize_with_aspect_ratio(
            weights, target_size=self._size, return_pad=True
        )
        # Zero out region outside the actual image after padding
        resized_weights = mask_padded_region(resized_weights, pad_coords)
        # Expand dim if needed
        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt, resized_weights

    def _try_crop_margin(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Attempt to do a margin-based crop multiple times.
        We define "margin" heuristics here by controlling how much foreground
        is in the crop, etc. (logic from original code).
        Returns None if no successful crop within attempt_limit.
        """
        height, width, _ = img.shape

        for _ in range(self._attempt_limit):
            # Random top-left inside the original image
            min_y = np.random.randint(0, height - self._size[0])
            min_x = np.random.randint(0, width - self._size[1])

            # Create a mask with 1 where we plan to crop
            trial_mask = np.zeros_like(tgt, dtype=np.int32)
            trial_mask[min_y : min_y + self._size[0], min_x : min_x + self._size[1]] = 1

            # Heuristic checks on how much "foreground" is inside or outside
            # In your original code: "if np.sum(mask * tgt) / area_in_crop > 0.8" => continue
            # We'll keep the same logic:
            area_in_crop = self._size[0] * self._size[1]
            overlap = np.sum(trial_mask * tgt) / float(area_in_crop)

            if overlap > 0.8 or overlap < 0.2:
                # Too much or too little foreground in the crop => skip
                continue

            # If we get here, we accept this crop
            crop_box = (min_y, min_x, self._size[0], self._size[1])
            return self._crop_and_resize(img, tgt, weights, crop_box)
        return None

    def _try_crop_bookmark(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Attempt to do a bookmark-based crop (guaranteeing some fraction of bookmark_label
        is in the crop).
        Returns None if no successful crop within attempt_limit.
        """
        height, width, _ = img.shape
        # Check if bookmark_label is even present
        bookmark_mask = tgt == self._bookmark_label
        bookmark_area = bookmark_mask.sum()
        if bookmark_area == 0:
            return None  # No bookmark => can't do this mode

        for _ in range(self._attempt_limit):
            min_y = np.random.randint(0, height - self._size[0])
            min_x = np.random.randint(0, width - self._size[1])

            # We'll see how much of the bookmark falls into this crop
            trial_mask = np.zeros_like(tgt, dtype=np.int32)
            trial_mask[min_y : min_y + self._size[0], min_x : min_x + self._size[1]] = 1

            overlap_bookmark = np.sum(trial_mask * bookmark_mask) / float(bookmark_area)

            # Original code used a 0.2 threshold
            if overlap_bookmark < 0.2:
                # Not enough bookmark => skip
                continue

            crop_box = (min_y, min_x, self._size[0], self._size[1])
            return self._crop_and_resize(img, tgt, weights, crop_box)
        return None

    def _try_random_scale_crop(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Do the random scale & ratio logic from the original code.
        We'll attempt it up to self._attempt_limit times.
        """
        height, width, _ = img.shape
        area = height * width

        for _ in range(self._attempt_limit):
            scale = np.random.uniform(self._scale[0], self._scale[1])
            ratio = np.random.uniform(self._ratio[0], self._ratio[1])

            new_area = area * scale
            new_height = int(round(math.sqrt(new_area * ratio)))
            new_width = int(round(math.sqrt(new_area / ratio)))

            if new_height >= height or new_width >= width:
                # Invalid, skip
                continue

            min_y = np.random.randint(0, height - new_height)
            min_x = np.random.randint(0, width - new_width)
            crop_box = (min_y, min_x, new_height, new_width)
            return self._crop_and_resize(img, tgt, weights, crop_box)

        # If all attempts fail, just do a no-crop fallback
        return self._no_crop(img, tgt, weights)

    def _crop_and_resize(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        crop_box: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crops img, tgt, weights using `crop_box` and then resizes them to self._size.
        """
        cropped_img, cropped_tgt, cropped_weights = crop_arrays(
            crop_box, img.copy(), tgt.copy(), weights.copy()
        )

        resized_img, _ = resize_with_aspect_ratio(cropped_img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(
            cropped_tgt, target_size=self._size, interpolation=cv2.INTER_NEAREST
        )

        resized_weights, _, pad_coords = resize_with_aspect_ratio(
            cropped_weights, target_size=self._size, return_pad=True
        )
        resized_weights = mask_padded_region(resized_weights, pad_coords)
        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt, resized_weights

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a random transformation (no-crop, margin-crop, bookmark-crop, or random-scale-crop)
        and then resize to self._size.

        Returns:
            (resized_img, resized_tgt, resized_weights)
        """
        # Decide which crop to do
        choice_prob = np.random.rand()

        # 1) Do not crop
        if choice_prob < self._not_crop_prob:
            return self._no_crop(img, tgt, weights)

        choice_prob -= self._not_crop_prob

        # 2) Crop with margin
        if choice_prob < self._crop_margin_prob:
            result = self._try_crop_margin(img, tgt, weights)
            if result is not None:
                return result
            # If margin crop failed after attempts, fallback to random scale crop

        choice_prob -= self._crop_margin_prob

        # 3) Crop with bookmark
        if choice_prob < self._crop_bookmark_prob and self._bookmark_label != -1:
            result = self._try_crop_bookmark(img, tgt, weights)
            if result is not None:
                return result
            # If bookmark crop failed after attempts, fallback to random scale crop

        # 4) Finally, random scale and ratio crop
        result = self._try_random_scale_crop(img, tgt, weights)
        return result


class Resize(object):
    def __init__(self, size: tuple[int] = (256, 256)):
        self._size = size

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        resized_img, _ = resize_with_aspect_ratio(img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(
            tgt, target_size=self._size, interpolation=cv2.INTER_NEAREST
        )
        resized_weights, _ = resize_with_aspect_ratio(weights, target_size=self._size)

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt, resized_weights


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
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img).float() / 255.0
        tgt = torch.from_numpy(tgt).float()
        weights = torch.from_numpy(weights).float()

        # permute img to (C, H, W)
        return img.permute(2, 0, 1), tgt, weights


class MaskToBinary(object):
    def __init__(self, foreground_label: int = 1, *args, **kwargs):
        super(MaskToBinary, self).__init__(*args, **kwargs)
        self._foreground_label = foreground_label

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # convert to binary mask
        # 0: background, 1: foreground
        tgt = (tgt == self._foreground_label).astype(np.int64)

        return img, tgt, weights


def build_transform():
    tr_fn = v2.Compose(
        [
            ImageToArray(normalize=False),
            RandomHorizontalFlip(),
            RandomResizedCrop(size=(1024, 1024)),
            MaskToBinary(),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_valid_transform():
    tr_fn = v2.Compose(
        [
            Resize(size=(1024, 1024)),
            MaskToBinary(foreground_label=2),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_scanned_transform():
    tr_fn = v2.Compose(
        [
            Rotate(),
            RandomHorizontalFlip(),
            RandomResizedCrop(
                size=(640, 640),
                scale=(0.25, 1.0),
                crop_margin_prob=0.3,
                crop_bookmark_prob=0.2,
                not_crop_prob=0.2,
                bookmark_label=1,
            ),
            MaskToBinary(foreground_label=2),
            ArrayToTensor(),
        ]
    )

    return tr_fn
