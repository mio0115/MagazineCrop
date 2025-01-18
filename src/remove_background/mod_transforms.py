import torch
from torchvision.transforms import v2
import torchvision.transforms as T
import numpy as np
import cv2

from ..utils.misc import resize_with_aspect_ratio


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
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a random rotation to the image, target mask, and weights.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights mask of shape (H, W).
            edge_len (float): The length of the edge of the image.
            edge_theta (float): The angle of the edge of the image.

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

        rotated_edge_theta = edge_theta + angle

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

        return rotated_img, rotated_tgt, rotated_weights, edge_len, rotated_edge_theta


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
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
            edge_theta = 180 - edge_theta

        return img, tgt, weights, edge_len, edge_theta


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
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Flips the input arrays vertically with (1 - not_flip_prob) chance.
        """
        # Draw a random number in [0,1). If it's >= not_flip_prob, we flip.
        if np.random.rand() >= self._not_flip_prob:
            img = np.flip(img, axis=0)
            tgt = np.flip(tgt, axis=0)
            weights = np.flip(weights, axis=0)
            edge_theta = 180 - edge_theta

        return img, tgt, weights, edge_len, edge_theta


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
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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

        return resized_img, resized_tgt, resized_weights, edge_len, edge_theta


class ArrayToTensor(object):
    """
    A transform that converts numpy arrays (image, target, and weights)
    into PyTorch tensors. It also scales the image to [0,1] by dividing by 255.
    """

    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if self._normalize:
            img = img / 255.0
        img = torch.as_tensor(img.copy(), dtype=torch.float32)
        tgt = torch.as_tensor(tgt.copy(), dtype=torch.float32)
        weights = torch.as_tensor(weights.copy(), dtype=torch.float32)
        edge_len = torch.tensor(edge_len, dtype=torch.float32)
        edge_theta = torch.tensor(edge_theta, dtype=torch.float32)

        # permute img to (C, H, W)
        return img.permute(2, 0, 1), tgt, weights, edge_len, edge_theta


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
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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

        return img, tgt, weights, edge_len, edge_theta


def build_valid_transform(size=(640, 640)):
    tr_fn = v2.Compose(
        [
            Resize(size=(640, 640)),
            MaskToBinary(foreground_label=2),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_scanned_transform():
    tr_fn = v2.Compose(
        [
            Rotate(random_angle_range=(-5, 5)),
            RandomHorizontalFlip(not_flip_prob=0.6),
            RandomVerticalFlip(not_flip_prob=0.9),
            Resize(size=(640, 640)),
            MaskToBinary(foreground_label=2),
            ArrayToTensor(),
        ]
    )

    return tr_fn


def build_inter_transform(split="train"):
    if split.lower() == "train":
        tr_fn = v2.Compose(
            [
                Rotate(random_angle_range=(-5, 5)),
                RandomHorizontalFlip(not_flip_prob=0.6),
                RandomVerticalFlip(not_flip_prob=0.9),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=False),
            ]
        )
    elif split.lower() == "valid":
        tr_fn = v2.Compose(
            [
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=False),
            ]
        )

    return tr_fn
