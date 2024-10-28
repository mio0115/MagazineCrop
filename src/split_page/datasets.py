import os
import json
from PIL import Image

import cv2
from torch.utils.data import Dataset

from .transforms import build_scanned_transforms


class MagazineCropDataset(Dataset):
    def __init__(self, split: str, augment_factor: int = 5, transforms=None):
        super(MagazineCropDataset, self).__init__()

        split = split.lower()
        if split not in ["train", "valid"]:
            raise ValueError(f"split must be either 'train' or 'valid', got {split}")

        self._path_to_root = os.path.join(
            os.getcwd(), "data", f"{split}_data", "scanned"
        )
        with open(
            os.path.join(self._path_to_root, "split_annotations.json"), "r"
        ) as fp_annotations:
            self._annotations = json.load(fp_annotations)

        self._keys = list(self._annotations.keys())
        self._orig_len = len(self._keys)
        self._augment_factor = augment_factor

        self._transforms = transforms

    def __len__(self):
        return self._augment_factor * self._orig_len

    def __getitem__(self, index):
        key = self._keys[index % self._orig_len]

        annotation = self._annotations[key]
        image = cv2.imread(
            os.path.join(self._path_to_root, annotation["imagePath"]),
        )
        # note that coordinates are in the form of (x, theta)
        # note that there could be more than 1
        coords = annotation["coordinates"][0]

        if self._transforms is not None:
            image, coords = self._transforms(image, coords)

        return image, coords


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from time import time

    ds = MagazineCropDataset(
        split="train", transforms=build_scanned_transforms(), augment_factor=1
    )

    x_coords = []

    start = time()
    for img, coords in DataLoader(ds, batch_size=1, num_workers=4):
        x_coords.append(coords[..., 0])
    end = time()
    print(f"Time taken: {end - start:.2f}s")

    mean_x = sum(x_coords) / len(x_coords)
    var_x = sum((x - mean_x) ** 2 for x in x_coords) / (len(x_coords) - 1)

    print(f"Mean x-coord: {mean_x}")
    print(f"Variance x-coord: {var_x}")
