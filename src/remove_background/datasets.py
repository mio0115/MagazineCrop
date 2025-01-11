import os
import json
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets

from ..utils.misc import polygon_to_mask


def resize_contour(contour, scale_factor, image_shape):
    """
    Resize a contour by a given scale factor.

    Args:
        contour (np.ndarray): Original contour points.
        scale_factor (float): Scale factor to resize the contour.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        np.ndarray: Resized contour.
    """
    # Compute the center of the contour
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Contour has zero area and cannot be resized.")
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    center = np.array([cx, cy])

    # Scale the contour
    resized_contour = (contour - center) * scale_factor + center

    # Clip the contour to ensure it's within the image boundaries
    resized_contour = np.clip(
        resized_contour, [0, 0], [image_shape[1] - 1, image_shape[0] - 1]
    )

    return resized_contour.astype(np.int32)


def set_weights_from_resized_contours(
    image_shape, contour, inner_ratio=0.9, outer_ratio=1.1
):
    """
    Generate a weight map based on resized contours.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        contour (np.ndarray): Contour representing the foreground.
        edge_weight (float): Weight for pixels close to the edge.
        inner_weight (float): Weight for pixels away from the edge.

    Returns:
        np.ndarray: Weight map with the same shape as the input image.
    """
    # Resize the contour
    smaller_contour = resize_contour(contour, inner_ratio, image_shape)
    larger_contour = resize_contour(contour, outer_ratio, image_shape)

    smaller_area, normal_area, larger_area = (
        cv2.contourArea(smaller_contour),
        cv2.contourArea(contour),
        cv2.contourArea(larger_contour),
    )
    outside_area = larger_area - normal_area

    rest_ratio = outside_area / (image_shape[0] * image_shape[1] - normal_area)
    edge_outside_ratio = 1 - rest_ratio
    edge_inner_ratio = 1 - smaller_area / normal_area
    edge_outer_ratio = 1 - edge_inner_ratio

    # Create masks for smaller and larger contours
    mask_smaller = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(
        mask_smaller, [smaller_contour], contourIdx=-1, color=1, thickness=-1
    )

    mask_normal = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask_normal, [contour], contourIdx=-1, color=1, thickness=-1)

    mask_larger = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(
        mask_larger, [larger_contour], contourIdx=-1, color=1, thickness=-1
    )

    # Create the weight map
    weight_map = np.full(image_shape, rest_ratio, dtype=np.float32)
    weight_map[mask_larger > 0] = edge_outside_ratio  # Edge region
    weight_map[mask_normal > 0] = edge_outer_ratio  # Outer region
    weight_map[mask_smaller > 0] = edge_inner_ratio  # Inner region

    return weight_map / 2


class MyVOCSegmentation(tv_datasets.VOCSegmentation):
    def __init__(self, augment_factor: int = 5, *args, **kwargs):
        super(MyVOCSegmentation, self).__init__(*args, **kwargs)

        self._orig_len = super().__len__()
        self._augment_factor = augment_factor

    def __len__(self):
        return self._augment_factor * super().__len__()

    def __getitem__(self, index):
        img, mask = super().__getitem__(index % self._orig_len)
        return img, mask


class MagazineCropDataset(Dataset):
    def __init__(self, split: str, augment_factor: int = 5, transforms=None):
        super(MagazineCropDataset, self).__init__()

        split = split.lower()
        if split not in ["train", "valid"]:
            raise ValueError(f"split must be either 'train' or 'valid', got {split}")

        # self._path_to_root = os.path.join(
        #     os.getcwd(), "data", f"{split}_data", "scanned"
        # )
        self._path_to_root = os.path.join(os.getcwd(), "data", "train_data", "scanned")
        with open(
            os.path.join(
                self._path_to_root, "annotations", f"{split}_annotations.json"
            ),
            "r",
        ) as fp_annotations:
            self._annotations = json.load(fp_annotations)

        self._keys = list(self._annotations.keys())
        self._orig_len = len(self._keys)
        self._augment_factor = augment_factor
        self._split = split

        self._labels = {
            "background": 0,
            "foreground": 1,
            "bookmark": 0,
            "bookmark_mask": 0,
        }
        self._label_order = ["bookmark", "bookmark_mask", "foreground"]

        self._transforms = transforms

    def __len__(self):
        return self._augment_factor * self._orig_len

    def _generate_labels_and_weights(self, polygons, height: int, width: int):
        if self._labels["background"] == 0:
            labels = np.zeros((height, width), dtype=np.uint8)
        else:
            labels = np.ones((height, width), dtype=np.uint8)

        cls_polygons = dict.fromkeys(self._label_order, [])
        for polygon in polygons:
            label = polygon["label"]
            if label not in self._label_order:
                continue

            cls_polygons[label].append(
                np.array(polygon["points"], dtype=np.int32).reshape(-1, 1, 2)
            )

        for curr_label in self._label_order:
            if len(cls_polygons[curr_label]) == 0:
                continue
            cv2.drawContours(
                image=labels,
                contours=cls_polygons[curr_label],
                contourIdx=-1,
                color=self._labels[curr_label],
                thickness=-1,
            )
        if self._split == "valid":
            weights = set_weights_from_resized_contours(
                image_shape=(height, width),
                contour=cls_polygons["foreground"][0],
                inner_ratio=0.9,
                outer_ratio=1.1,
            )
        else:
            weights = np.ones_like(labels)

        return labels, weights

    def __getitem__(self, index):
        key = self._keys[index % self._orig_len]

        annotation = self._annotations[key]

        pil_image = Image.open(
            os.path.join(self._path_to_root, annotation["imagePath"])
        ).convert("RGB")
        image = np.array(pil_image, dtype=np.uint8)[..., ::-1].copy()

        labels, weights = self._generate_labels_and_weights(
            polygons=annotation["shapes"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        if self._transforms is not None:
            image, labels, weights = self._transforms(image, labels, weights)

        return image, labels, weights


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .transforms import build_scanned_transform

    ds = MagazineCropDataset(split="train", transforms=build_scanned_transform())
    dl = DataLoader(ds, batch_size=1, num_workers=4)

    for img, labels, weights in dl:
        continue
