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


def mc_collate_fn(batch):
    images, labels, weights = zip(*batch)

    inputs = {
        "images": torch.stack(images, dim=0),
    }
    targets = {
        "labels": torch.stack(labels, dim=0),
    }
    weights = torch.stack(weights, dim=0)
    return inputs, targets, weights


class MagazineCropDataset(Dataset):
    def __init__(self, split: str, augment_factor: int = 5, transforms=None):
        super(MagazineCropDataset, self).__init__()

        split = split.lower()
        if split not in ["train", "valid"]:
            raise ValueError(f"split must be either 'train' or 'valid', got {split}")

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
            "foreground": 2,
            "bookmark": 1,
            "bookmark_mask": 0,
        }
        self._label_order = ["bookmark", "bookmark_mask", "foreground"]

        self._transforms = transforms

    def __len__(self):
        return self._augment_factor * self._orig_len

    def _generate_labels_and_weights(self, polygons, height: int, width: int):
        labels = np.zeros((height, width), dtype=np.uint8)

        cls_polygons = {k: [] for k in self._label_order}
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
        orig_image = np.array(pil_image, dtype=np.uint8)[..., ::-1].copy()
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)[..., None]

        image = np.concatenate([orig_image, gray_image], axis=-1)

        labels, weights = self._generate_labels_and_weights(
            polygons=annotation["shapes"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        if self._transforms is not None:
            image, labels, weights = self._transforms(orig_image, labels, weights)

        return image, labels, weights


def mod_mc_collate_fn(batch):
    images, labels, weights, edge_lens, edge_thetas, edges = zip(*batch)

    inputs = {
        "images": torch.stack(images, dim=0),
        "length": torch.stack(edge_lens, dim=0),
        "theta": torch.stack(edge_thetas, dim=0),
    }
    targets = {
        "labels": torch.stack(labels, dim=0),
        "corner_coordinates": torch.stack(edges, dim=0),
    }
    weights = torch.stack(weights, dim=0)
    return inputs, targets, weights


class ModMagazineCropDataset(Dataset):
    def __init__(
        self, split: str, augment_factor: int = 5, transforms=None, edge_size: int = 640
    ):
        super(ModMagazineCropDataset, self).__init__()

        split = split.lower()
        if split not in ["train", "valid"]:
            raise ValueError(f"split must be either 'train' or 'valid', got {split}")

        self._path_to_root = os.path.join(os.getcwd(), "data", "train_data", "scanned")
        with open(
            os.path.join(
                self._path_to_root, "annotations", f"{split}_annotations.json"
            ),
            "r",
        ) as fp_annotations:
            self._annotations = json.load(fp_annotations)
        with open(
            os.path.join(self._path_to_root, "annotations", f"edge_annotations.json"),
            "r",
        ) as fp_annotations:
            self._edge_annotations = json.load(fp_annotations)

        self._keys = list(self._annotations.keys())
        self._orig_len = len(self._keys)
        self._augment_factor = augment_factor
        self._split = split

        self._labels = {
            "background": 0,
            "foreground": 2,
            "bookmark": 1,
            "bookmark_mask": 0,
        }
        self._label_order = ["bookmark", "bookmark_mask", "foreground"]

        self._transforms = transforms

    def __len__(self):
        return self._augment_factor * self._orig_len

    def _generate_labels_and_weights(self, polygons, height: int, width: int):
        labels = np.zeros((height, width), dtype=np.uint8)

        cls_polygons = {k: [] for k in self._label_order}
        edges = []
        for polygon in polygons:
            label = polygon["label"]
            if label not in self._label_order:
                if label == "edge":
                    edge = np.array(polygon["points"], dtype=np.float32).reshape(-1, 2)
                    sorted_edge = edge[np.argsort(edge[:, 1])]
                    edges.append(sorted_edge)
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

        edges = np.array(edges, dtype=np.float32)
        edges = edges[np.argsort(edges[:, 0, 0])].reshape(-1, 2)

        return (
            labels,
            weights,
            edges,
        )

    def __getitem__(self, index):
        key = self._keys[index % self._orig_len]

        annotation = self._annotations[key]
        image_dir = annotation["imagePath"].split(os.sep)[1]
        edge_annotation = self._edge_annotations[image_dir]

        pil_image = Image.open(
            os.path.join(self._path_to_root, annotation["imagePath"])
        ).convert("RGB")
        orig_image = np.flip(np.array(pil_image, dtype=np.uint8), -1)
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)[..., None]

        image = np.concatenate([orig_image, gray_image], axis=-1)

        labels, weights, edges = self._generate_labels_and_weights(
            polygons=annotation["shapes"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        # print(orig_image.shape)
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("image", 1024, 1024)

        # cv2.line(orig_image, tuple(edges[0]), tuple(edges[1]), (0, 255, 0), 2)
        # cv2.line(orig_image, tuple(edges[2]), tuple(edges[3]), (0, 0, 255), 2)
        # cv2.imshow("image", orig_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        edge_len, edge_theta = edge_annotation["edge_length"], edge_annotation["theta"]
        if self._transforms is not None:
            image, labels, weights, edge_len, edge_theta, edges = self._transforms(
                image, labels, weights, edge_len, edge_theta, edges
            )

        return image, labels, weights, edge_len, edge_theta, edges


class InterMagazineCropDataset(Dataset):
    def __init__(self, split: str, augment_factor: int = 5, transforms=None):
        super(InterMagazineCropDataset, self).__init__()

        split = split.lower()
        if split not in ["train", "valid"]:
            raise ValueError(f"split must be either 'train' or 'valid', got {split}")

        self._path_to_root = os.path.join(os.getcwd(), "data", "train_data", "scanned")
        with open(
            os.path.join(
                self._path_to_root, "annotations", f"{split}_annotations.json"
            ),
            "r",
        ) as fp_annotations:
            self._annotations = json.load(fp_annotations)
        with open(
            os.path.join(
                self._path_to_root, "annotations", "edge_annotations_640.json"
            ),
            "r",
        ) as fp_annotations:
            self._edge_annotations = json.load(fp_annotations)

        self._keys = list(self._annotations.keys())
        self._orig_len = len(self._keys)
        self._augment_factor = augment_factor
        self._split = split

        self._labels = {
            "background": 0,
            "foreground": 2,
            "bookmark": 1,
            "bookmark_mask": 0,
        }
        self._label_order = ["bookmark", "bookmark_mask", "foreground"]

        self._transforms = transforms

    def __len__(self):
        return self._augment_factor * self._orig_len

    def _generate_labels_and_weights(self, polygons, height: int, width: int):
        labels = np.zeros((height, width), dtype=np.uint8)

        cls_polygons = {k: [] for k in self._label_order}
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
        image_dir = annotation["imagePath"].split(os.sep)[1]
        edge_annotation = self._edge_annotations[image_dir]
        image_name = os.path.basename(annotation["imagePath"]).split(".")[0]
        inter_image = np.load(
            os.path.join(
                self._path_to_root, "intermediate", image_dir, f"{image_name}.npy"
            )
        )
        inter_image = np.squeeze(inter_image)[..., None]

        labels, weights = self._generate_labels_and_weights(
            polygons=annotation["shapes"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        edge_len, edge_theta = edge_annotation["edge_length"], edge_annotation["theta"]
        if self._transforms is not None:
            image, labels, weights, edge_len, edge_theta = self._transforms(
                inter_image, labels, weights, edge_len, edge_theta
            )

        return image, labels, weights, edge_len, edge_theta


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .mod_transforms import build_inter_transform

    ds = InterMagazineCropDataset(
        split="train", transforms=build_inter_transform(split="train"), augment_factor=5
    )
    dl = DataLoader(ds, batch_size=8, num_workers=4)

    bytes_per_element = torch.tensor([], dtype=torch.float32).element_size()

    for img, labels, weights, edge_len, edge_theta in dl:
        vram_usage_per_batch = img.flatten().shape[0] * bytes_per_element

        print(f"VRAM usage per batch: {vram_usage_per_batch / 1e9:.2f} GB")
    print("Done")
