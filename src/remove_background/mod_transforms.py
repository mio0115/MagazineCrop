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

    def rotate_points(
        self, points: np.ndarray, angle: float, center: np.ndarray
    ) -> np.ndarray:
        """
        Rotate points by a given angle around the center.

        Args:
            points (np.ndarray): An array of shape (N, 2) representing N points.
            angle (float): The angle in degrees to rotate the points.
            center (np.ndarray): The center point of the rotation.

        Returns:
            np.ndarray: The rotated points.
        """
        # Shift the points so that the center is at the origin.
        shifted = points - center
        # Convert the angle to radians.
        theta = np.deg2rad(angle)
        # Create the rotation matrix.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Apply the rotation matrix to the points.
        rotated_x = shifted[:, 0] * cos_t - shifted[:, 1] * sin_t
        rotated_y = shifted[:, 0] * sin_t + shifted[:, 1] * cos_t

        rotated_shifted = np.stack([rotated_x, rotated_y], axis=1)

        return rotated_shifted + center

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
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
        rotated_edges = self.rotate_points(
            edges, -angle, np.array(center, dtype=np.float32).reshape(1, -1)
        )

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

        # cv2.namedWindow("rotated_img", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("rotated_img", 1024, 1024)

        # print(edge_theta, rotated_edge_theta)
        # tmp_img = rotated_img.copy()
        # print(np.reshape(rotated_edges, shape=(2, 2, 2)).shape)
        # for edge in np.reshape(rotated_edges, shape=(2, 2, 2)):
        #     tmp_edge = edge.astype(np.int32)
        #     cv2.line(tmp_img, tmp_edge[0], tmp_edge[1], (0, 255, 0), 2)
        # cv2.imshow("rotated_img", tmp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # If the original image was (H, W, 1) and after rotation is (H, W),
        # we add back a dummy channel to match dimensions if needed.
        if img.ndim > rotated_img.ndim:
            rotated_img = rotated_img[..., None]

        return (
            rotated_img,
            rotated_tgt,
            rotated_weights,
            edge_len,
            rotated_edge_theta,
            rotated_edges,
        )


class RandomShift:
    """
    Randomly shifts an image, mask, and weights array if there is foreground
    near the edges, moving the content in one of four directions: top, bottom,
    left, or right.
    """

    def __init__(self, not_shift_prob: float = 0.5, background_label: int = 0):
        """
        Args:
            not_shift_prob (float): Probability of NOT performing a shift.
            background_label (int): Label for background in the target mask.
        """
        self.not_shift_prob = not_shift_prob
        self.background_label = background_label

    def _foreground_bbox(self, tgt: np.ndarray) -> tuple[int, int, int, int]:
        """
        Finds the bounding box of the foreground in 'tgt'.
        Returns (top, bottom, left, right).
        If the entire mask is background, returns (0, H-1, 0, W-1).
        """
        height, width = tgt.shape[:2]

        top = 0
        while top < height and tgt[top, :].max() == self.background_label:
            top += 1

        bottom = height - 1
        while bottom >= 0 and tgt[bottom, :].max() == self.background_label:
            bottom -= 1

        left = 0
        while left < width and tgt[:, left].max() == self.background_label:
            left += 1

        right = width - 1
        while right >= 0 and tgt[:, right].max() == self.background_label:
            right -= 1

        # If entire mask is background, clamp to full image.
        if top >= bottom or left >= right:
            return (0, height - 1, 0, width - 1)
        return (top, bottom, left, right)

    def _shift_up(self, img, tgt, weights, edges, shift):
        """
        Shift image 'up' by 'shift' rows.
        """
        h, w = img.shape[:2]
        # Assume img is (H, W, C)
        img = np.pad(
            img[shift:, :, :],
            pad_width=((0, shift), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        tgt = np.pad(
            tgt[shift:, :],
            pad_width=((0, shift), (0, 0)),
            mode="constant",
            constant_values=self.background_label,
        )
        weights = np.pad(
            weights[shift:, :],
            pad_width=((0, shift), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # Move edges up
        edges[:, 1] -= shift
        return img, tgt, weights, edges

    def _shift_down(self, img, tgt, weights, edges, shift):
        """
        Shift image 'down' by 'shift' rows.
        """
        h, w = img.shape[:2]
        img = np.pad(
            img[:-shift, :, :],
            pad_width=((shift, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        tgt = np.pad(
            tgt[:-shift, :],
            pad_width=((shift, 0), (0, 0)),
            mode="constant",
            constant_values=self.background_label,
        )
        weights = np.pad(
            weights[:-shift, :],
            pad_width=((shift, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        edges[:, 1] += shift
        return img, tgt, weights, edges

    def _shift_left(self, img, tgt, weights, edges, shift):
        """
        Shift image 'left' by 'shift' columns.
        """
        h, w = img.shape[:2]
        img = np.pad(
            img[:, shift:, :],
            pad_width=((0, 0), (0, shift), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        tgt = np.pad(
            tgt[:, shift:],
            pad_width=((0, 0), (0, shift)),
            mode="constant",
            constant_values=self.background_label,
        )
        weights = np.pad(
            weights[:, shift:],
            pad_width=((0, 0), (0, shift)),
            mode="constant",
            constant_values=0,
        )
        edges[:, 0] -= shift
        return img, tgt, weights, edges

    def _shift_right(self, img, tgt, weights, edges, shift):
        """
        Shift image 'right' by 'shift' columns.
        """
        h, w = img.shape[:2]
        img = np.pad(
            img[:, :-shift, :],
            pad_width=((0, 0), (shift, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        tgt = np.pad(
            tgt[:, :-shift],
            pad_width=((0, 0), (shift, 0)),
            mode="constant",
            constant_values=self.background_label,
        )
        weights = np.pad(
            weights[:, :-shift],
            pad_width=((0, 0), (shift, 0)),
            mode="constant",
            constant_values=0,
        )
        edges[:, 0] += shift
        return img, tgt, weights, edges

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        """
        Possibly shifts the foreground content up/down/left/right if
        there is background border. Probability of shift is (1 - not_shift_prob).
        """
        h, w = img.shape[:2]
        top, bottom, left, right = self._foreground_bbox(tgt)
        top_space = top
        bottom_space = (h - 1) - bottom
        left_space = left
        right_space = (w - 1) - right

        # If the entire mask is foreground, or no background border around it,
        # or random says do not shift => skip shifting
        if (
            top_space == 0
            and bottom_space == 0
            and left_space == 0
            and right_space == 0
        ) or (np.random.rand() < self.not_shift_prob):
            return img, tgt, weights, edge_len, edge_theta, edges

        # Decide vertical shift direction
        top_bottom_choice = None
        if top_space > 0 or bottom_space > 0:
            if top_space == 0:
                top_bottom_choice = 1  # must shift down
            elif bottom_space == 0:
                top_bottom_choice = 0  # must shift up
            else:
                top_bottom_choice = np.random.choice(
                    [0, 1],
                    p=[
                        top_space / (top_space + bottom_space),
                        bottom_space / (top_space + bottom_space),
                    ],
                )  # up or down

        # Decide horizontal shift direction
        left_right_choice = None
        if left_space > 0 or right_space > 0:
            if left_space == 0:
                left_right_choice = 1  # must shift right
            elif right_space == 0:
                left_right_choice = 0  # must shift left
            else:
                left_right_choice = np.random.choice(
                    [0, 1],
                    p=[
                        left_space / (left_space + right_space),
                        right_space / (left_space + right_space),
                    ],
                )  # left or right

        # Execute vertical shift
        if top_bottom_choice == 0:
            # shift up
            shift_x = np.random.randint(top_space // 4, top_space + 1)
            img, tgt, weights, edges = self._shift_up(img, tgt, weights, edges, shift_x)
        elif top_bottom_choice == 1:
            # shift down
            shift_x = np.random.randint(bottom_space // 4, bottom_space + 1)
            img, tgt, weights, edges = self._shift_down(
                img, tgt, weights, edges, shift_x
            )

        # Execute horizontal shift
        if left_right_choice == 0:
            # shift left
            shift_y = np.random.randint(left_space // 4, left_space + 1)
            img, tgt, weights, edges = self._shift_left(
                img, tgt, weights, edges, shift_y
            )
        elif left_right_choice == 1:
            # shift right
            shift_y = np.random.randint(right_space // 4, right_space + 1)
            img, tgt, weights, edges = self._shift_right(
                img, tgt, weights, edges, shift_y
            )

        # Finally, clamp edges
        edges[:, 0] = np.clip(edges[:, 0], 0, w - 1)
        edges[:, 1] = np.clip(edges[:, 1], 0, h - 1)

        # cv2.namedWindow("shifted img", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("shifted img", 1024, 1024)

        # tmp_img = img.copy()
        # print(np.reshape(edges, shape=(2, 2, 2)).shape)
        # for edge in np.reshape(edges, shape=(2, 2, 2)):
        #     tmp_edge = edge.astype(np.int32)
        #     cv2.line(tmp_img, tmp_edge[0], tmp_edge[1], (0, 255, 0), 2)
        # cv2.imshow("shifted img", tmp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img, tgt, weights, edge_len, edge_theta, edges


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
        self._flipped_order = [2, 3, 0, 1]

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
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
        width = img.shape[1]
        # Draw a random number in [0,1). If it's >= not_flip_prob, we flip.
        if np.random.rand() >= self._not_flip_prob:
            # Flip along axis=1 (left-right flip).
            img = np.flip(img, axis=1)
            tgt = np.flip(tgt, axis=1)
            weights = np.flip(weights, axis=1)
            edge_theta = 180 - edge_theta

            edges[:, 0] = width - edges[:, 0]
            # arange the edges
            edges = edges[self._flipped_order]

        return img, tgt, weights, edge_len, edge_theta, edges


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
        self._flipped_order = [1, 0, 3, 2]

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        """
        Flips the input arrays vertically with (1 - not_flip_prob) chance.
        """
        height = img.shape[0]
        # Draw a random number in [0,1). If it's >= not_flip_prob, we flip.
        if np.random.rand() >= self._not_flip_prob:
            img = np.flip(img, axis=0)
            tgt = np.flip(tgt, axis=0)
            weights = np.flip(weights, axis=0)
            edge_theta = 180 - edge_theta

            edges[:, 1] = height - edges[:, 1]
            # Reorder the edges after flipping
            edges = edges[self._flipped_order]

        return img, tgt, weights, edge_len, edge_theta, edges


class Resize(object):
    """
    A transform that resizes the image, target mask, and weights array
    to a specified size (height, width) while preserving aspect ratio
    via a custom resize function.
    """

    def __init__(
        self,
        size: tuple[int] = (256, 256),
        resize_img: bool = True,
        resize_tgt: bool = True,
        resize_weights: bool = True,
    ):
        """
        Args:
            size (tuple[int, int]): The target (width, height) for resizing.
        """
        self._size = size
        self._resize_img = resize_img
        self._resize_tgt = resize_tgt
        self._resize_weights = resize_weights

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        """
        Resizes the image, target mask, and weights to self._size.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights mask array of shape (H, W).
            edge_len (float): The length of the edge of the image.
            edge_theta (float): The angle of the edge of the image.
            edges (np.ndarray): The edge points of the image.

        Returns:
            (resized_img, resized_tgt, resized_weights, edge_len, edge_theta, edges): Arrays resized to self._size.
        """
        if self._resize_img:
            resized_img, _, paddings = resize_with_aspect_ratio(
                img, target_size=self._size, return_pad=True
            )
            # Also compute corresponding coordinates of edges in resized image
            top, bottom, left, right = paddings
            gt_size = (self._size[0] - (left + right), self._size[1] - (top + bottom))
            edges[:, 0] = edges[:, 0] * gt_size[0] / img.shape[1] + left
            edges[:, 1] = edges[:, 1] * gt_size[1] / img.shape[0] + top
        else:
            resized_img = img
        if self._resize_tgt:
            resized_tgt, _ = resize_with_aspect_ratio(
                tgt, target_size=self._size, interpolation=cv2.INTER_NEAREST
            )
        else:
            resized_tgt = tgt
        if self._resize_weights:
            resized_weights, _ = resize_with_aspect_ratio(
                weights, target_size=self._size
            )
        else:
            resized_weights = weights

        if img.ndim > resized_img.ndim:
            resized_img = resized_img[..., None]

        return resized_img, resized_tgt, resized_weights, edge_len, edge_theta, edges


class ArrayToTensor(object):
    """
    A transform that converts numpy arrays (image, target, and weights)
    into PyTorch tensors. It also scales the image to [0,1] by dividing by 255.
    """

    def __init__(self, normalize=True, size: tuple[int, int] = (1024, 1024)):
        self._normalize = normalize
        self._size = size  # (height, width)

    def __call__(
        self,
        img: np.ndarray,
        tgt: np.ndarray,
        weights: np.ndarray,
        edge_len: float,
        edge_theta: float,
        edges: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img (np.ndarray): Image array of shape (H, W, C) in [0..255].
            tgt (np.ndarray): Target mask array of shape (H, W).
            weights (np.ndarray): Weights array of shape (H, W).
            edge_len (float): The length of the edge of the image.
            edge_theta (float): The angle of the edge of the image.
            edges (np.ndarray): The edge points of the image.

        Returns:
            (img_tensor, tgt_tensor, weights_tensor, edge_len, edge_theta, edges):
                - img_tensor shape: (C, H, W), float32 in [0, 1].
                - tgt_tensor shape: (H, W), float32.
                - weights_tensor shape: (H, W), float32.
                - edge_len shape: (1), float32.
                - edge_theta shape: (1), float32.
                - edges shape: (N, 2), float32.
        """
        # Convert from numpy arrays to PyTorch tensors
        if self._normalize:
            img = img / 255.0
        img = torch.as_tensor(img.copy(), dtype=torch.float32)
        tgt = torch.as_tensor(tgt.copy(), dtype=torch.float32)
        weights = torch.as_tensor(weights.copy(), dtype=torch.float32)
        edge_len = torch.tensor(edge_len, dtype=torch.float32)

        edge_radius = np.deg2rad(edge_theta) / np.pi
        edge_theta = torch.tensor(edge_radius, dtype=torch.float32)

        edges = torch.tensor(edges, dtype=torch.float32)
        edges[..., 0] = edges[..., 0] / self._size[1]
        edges[..., 1] = edges[..., 1] / self._size[0]

        # permute img to (C, H, W)
        return img.permute(2, 0, 1), tgt, weights, edge_len, edge_theta, edges


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
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Args:
            img (np.ndarray): Image array of shape (H, W, C) or (H, W).
            tgt (np.ndarray): Target mask with integer labels, shape (H, W).
            weights (np.ndarray): Weights array, shape (H, W).
            edge_len (float): The length of the edge of the image.
            edge_theta (float): The angle of the edge of the image.
            edges (np.ndarray): The edge points of the image.

        Returns:
            (img, binary_tgt, weights, edge_len, edge_theta, edges):
                - The original image unchanged.
                - A binary mask (0 or 1) where 1 indicates `foreground_label` pixels.
                - The original weights unchanged.
                - The original edge_len unchanged.
                - The original edge_theta unchanged.
                - The original edges unchanged.
        """
        # Convert to binary mask
        # Set to 1 if the target is the foreground label, 0 otherwise
        tgt = (tgt == self._foreground_label).astype(np.int64)

        return img, tgt, weights, edge_len, edge_theta, edges


def build_scanned_transform(split="train", size: tuple[int, int] = (1024, 1024)):
    if split.lower() == "train":
        tr_fn = v2.Compose(
            [
                Rotate(random_angle_range=(-5, 5)),
                RandomHorizontalFlip(not_flip_prob=0.5),
                RandomVerticalFlip(not_flip_prob=0.9),
                RandomShift(not_shift_prob=0.25, background_label=0),
                Resize(size=size),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=True, size=size),
            ]
        )
    elif split.lower() == "valid":
        tr_fn = v2.Compose(
            [
                Resize(size=size),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=True, size=size),
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
                Resize(size=(640, 640), resize_img=False),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=False),
            ]
        )
    elif split.lower() == "valid":
        tr_fn = v2.Compose(
            [
                Resize(size=(640, 640), resize_img=False),
                MaskToBinary(foreground_label=2),
                ArrayToTensor(normalize=False),
            ]
        )

    return tr_fn
