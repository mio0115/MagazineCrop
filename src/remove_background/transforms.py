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
    """
    A transform that rotates the input image, target mask, and weights
    by a random angle within a specified range.
    """

    def __init__(self, random_angle_range=(-20, 20)):
        """
        Args:
            random_angle_range (tuple[int, int]): A tuple (min_angle, max_angle)
                specifying the inclusive range of angles in degrees from which
                we randomly choose for rotation.
        """
        self._random_angle_range = random_angle_range

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a random rotation to the image, target mask, and weights.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights mask of shape (H, W).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The rotated image,
            target, and weights arrays, with the same shapes as the inputs.
        """
        # Pick a random rotation angle (in degrees) from the specified range.
        angle = np.random.randint(*self._random_angle_range)

        # Get image dimensions and compute center (for rotation pivot).
        height, width = img.shape[:2]
        center = (width // 2, height // 2)

        # Create a rotation matrix around the center with the specified angle.
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image with (intersection_x, height//2) as the center
        rotated_img = cv2.warpAffine(img, rot_mat, (width, height))

        # Rotate the target mask with nearest-neighbor to preserve class labels.
        rotated_tgt = cv2.warpAffine(
            tgt, rot_mat, (width, height), flags=cv2.INTER_NEAREST
        )

        # Rotate the weights mask with area interpolation to preserve weights.
        rotated_weights = cv2.warpAffine(
            weights, rot_mat, (width, height), flags=cv2.INTER_AREA
        )

        # If the original image was (H, W, 1) and after rotation is (H, W),
        # we add back a dummy channel to match dimensions if needed.
        if img.ndim > rotated_img.ndim:
            rotated_img = rotated_img[..., None]

        return rotated_img, rotated_tgt, rotated_weights


class RandomHorizontalFlip(object):
    """
    A transform that randomly flips the image, target mask, and weights
    horizontally (left-right) with a given probability.
    """

    def __init__(self, not_flip_prob=0.5):
        """
        Args:
            not_flip_prob (float): The probability of NOT flipping the image.
                If a random number is >= this value, we perform the flip.
        """
        self._not_flip_prob = not_flip_prob

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flips the input arrays horizontally with (1 - not_flip_prob) chance.

        Args:
            img (np.ndarray): Input image array.
            tgt (np.ndarray): Target mask array.
            weights (np.ndarray): Weights mask array.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The flipped (or unchanged)
            image, target, and weights arrays.
        """
        # Draw a random number in [0,1). If it's >= not_flip_prob, we flip.
        if np.random.rand() >= self._not_flip_prob:
            # Flip along axis=1 (left-right flip).
            img = np.flip(img, axis=1)
            tgt = np.flip(tgt, axis=1)
            weights = np.flip(weights, axis=1)

        return img, tgt, weights


class RandomVerticalFlip(object):
    """
    A transform that randomly flips the image, target mask, and weights
    vertically (up-down) with a given probability.
    """

    def __init__(self, not_flip_prob=0.5):
        """
        Args:
            not_flip_prob (float): The probability of NOT flipping the image.
                If a random number is >= this value, we perform the flip.
        """
        self._not_flip_prob = not_flip_prob

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flips the input arrays vertically with (1 - not_flip_prob) chance.
        """
        # Draw a random number in [0,1). If it's >= not_flip_prob, we flip.
        if np.random.rand() >= self._not_flip_prob:
            img = np.flip(img, axis=0)
            tgt = np.flip(tgt, axis=0)
            weights = np.flip(weights, axis=0)

        return img, tgt, weights


class PrintShape(object):
    """
    A debugging transform that prints the shapes of the image and target
    during the pipeline.
    """

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prints the shapes of the given arrays, then returns them unchanged.

        Args:
            img (np.ndarray): Image array.
            tgt (np.ndarray): Target mask array.

        Returns:
            tuple[np.ndarray, np.ndarray]: The same (img, tgt).
        """
        print(
            f"image shape: {img.shape}, target shape: {tgt.shape}, weights shape: {weights.shape}"
        )
        return img, tgt, weights


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
            scale = np.random.uniform(low=1.5, high=3.0)
            crop_size = tuple([int(l * scale) for l in self._size])
            # Random top-left inside the original image
            min_y = np.random.randint(0, height - crop_size[0])
            min_x = np.random.randint(0, width - crop_size[1])

            # Create a mask with 1 where we plan to crop
            trial_mask = np.zeros_like(tgt, dtype=np.int32)
            trial_mask[min_y : min_y + crop_size[0], min_x : min_x + crop_size[1]] = 1

            # Heuristic checks on how much "foreground" is inside or outside
            # In your original code: "if np.sum(mask * tgt) / area_in_crop > 0.8" => continue
            # We'll keep the same logic:
            area_in_crop = crop_size[0] * crop_size[1]
            overlap = np.sum(trial_mask * tgt) / float(area_in_crop)

            if overlap > 0.8 or overlap < 0.2:
                # Too much or too little foreground in the crop => skip
                continue

            # If we get here, we accept this crop
            crop_box = (min_y, min_x, crop_size[0], crop_size[1])
            return self._crop_and_resize(img, tgt, weights, crop_box, where="margin")
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
            scale = np.random.uniform(low=1.0, high=2.5)
            crop_size = tuple([int(l * scale) for l in self._size])

            min_y = np.random.randint(0, height - crop_size[0])
            min_x = np.random.randint(0, width - crop_size[1])

            # We'll see how much of the bookmark falls into this crop
            trial_mask = np.zeros_like(tgt, dtype=np.int32)
            trial_mask[min_y : min_y + crop_size[0], min_x : min_x + crop_size[1]] = 1

            overlap_bookmark = np.sum(trial_mask * bookmark_mask) / float(bookmark_area)

            # Original code used a 0.2 threshold
            if overlap_bookmark < 0.2:
                # Not enough bookmark => skip
                continue

            crop_box = (min_y, min_x, crop_size[0], crop_size[1])
            return self._crop_and_resize(img, tgt, weights, crop_box, where="bookmark")
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
            return self._crop_and_resize(
                img, tgt, weights, crop_box, where="random_scale"
            )

        # If all attempts fail, just do a no-crop fallback
        return self._no_crop(img, tgt, weights)

    def _crop_and_resize(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        crop_box: tuple[int, int, int, int],
        where: str = "bookmark",
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
    """
    A transform that resizes the image, target mask, and weights array
    to a specified size (height, width) while preserving aspect ratio
    via a custom resize function.
    """

    def __init__(self, size: tuple[int] = (256, 256)):
        """
        Args:
            size (tuple[int, int]): The target (height, width) for resizing.
        """
        self._size = size

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resizes the image, target mask, and weights to self._size.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights mask array of shape (H, W).

        Returns:
            (resized_img, resized_tgt, resized_weights): Arrays resized to self._size.
        """
        resized_img, _ = resize_with_aspect_ratio(img, target_size=self._size)
        resized_tgt, _ = resize_with_aspect_ratio(
            tgt, target_size=self._size, interpolation=cv2.INTER_NEAREST
        )
        resized_weights, _ = resize_with_aspect_ratio(weights, target_size=self._size)

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt, resized_weights


class CenterCrop(object):
    """
    A transform that performs a center crop of the image and target mask
    to a specified (height, width).
    """

    def __init__(self, size: tuple[int] = (224, 224)):
        """
        Args:
            size (tuple[int, int]): The (height, width) for the cropped output.
        """
        self._size = size

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crops the image and target mask at the center to self._size.

        Args:
            img (np.ndarray): Input image array of shape (H, W, C).
            tgt (np.ndarray): Target mask array of shape (H, W).

        Returns:
            (cropped_img, cropped_tgt): Center-cropped arrays
            of shapes (th, tw, C) and (th, tw) respectively.
        """
        h, w, _ = img.shape
        th, tw = self._size

        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))

        return img[i : i + th, j : j + tw], tgt[i : i + th, j : j + tw]


class ArrayToTensor(object):
    """
    A transform that converts numpy arrays (image, target, and weights)
    into PyTorch tensors. It also scales the image to [0,1] by dividing by 255.
    """

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img (np.ndarray): Image array of shape (H, W, C) in [0..255].
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights array of shape (H, W).

        Returns:
            (img_tensor, tgt_tensor, weights_tensor):
                - img_tensor shape: (C, H, W), float32 in [0, 1].
                - tgt_tensor shape: (H, W), float32.
                - weights_tensor shape: (H, W), float32.
        """
        # Convert from numpy arrays to PyTorch tensors
        img = torch.from_numpy(img).float() / 255.0
        tgt = torch.from_numpy(tgt).float()
        weights = torch.from_numpy(weights).float()

        # permute img to (C, H, W)
        return img.permute(2, 0, 1), tgt, weights


class MaskToBinary(object):
    """
    A transform that converts the target mask from multi-class or
    multi-label into a binary (foreground vs. background) mask.
    """

    def __init__(self, foreground_label: int = 1, *args, **kwargs):
        """
        Args:
            foreground_label (int): The class ID that should be considered 'foreground'.
        """
        super(MaskToBinary, self).__init__(*args, **kwargs)
        self._foreground_label = foreground_label

    def __call__(
        self, img: np.ndarray, tgt: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img (np.ndarray): Image array of shape (H, W, C) or (H, W).
            tgt (np.ndarray): Target mask with integer labels, shape (H, W).
            weights (np.ndarray): Weights array, shape (H, W).

        Returns:
            (img, binary_tgt, weights):
                - The original image unchanged.
                - A binary mask (0 or 1) where 1 indicates `foreground_label` pixels.
                - The original weights unchanged.
        """
        # Convert to binary mask
        # Set to 1 if the target is the foreground label, 0 otherwise
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


def build_scanned_transform(split="train", reshape_size: int = 1024):
    if split.lower() == "train":
        tr_fn = v2.Compose(
            [
                Rotate(),
                RandomHorizontalFlip(not_flip_prob=0.6),
                RandomVerticalFlip(not_flip_prob=0.9),
                RandomResizedCrop(
                    size=(reshape_size, reshape_size),
                    scale=(0.25, 1.0),
                    not_crop_prob=0.1,
                    crop_margin_prob=0.3,
                    crop_bookmark_prob=0.3,
                    bookmark_label=1,
                ),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(),
            ]
        )
    else:
        tr_fn = v2.Compose(
            [
                Resize(size=(reshape_size, reshape_size)),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(),
            ]
        )

    return tr_fn
