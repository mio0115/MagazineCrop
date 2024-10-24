import os
import json
from PIL import Image

import cv2
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets

from .transforms import build_scanned_transforms
from ..utils.misc import line_to_coord


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
            os.path.join(self._path_to_root, "page_split_annotations.json"), "r"
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
            cv2.IMREAD_GRAYSCALE,
        )
        coord = line_to_coord(
            polygon=annotation["shapes"][0]["points"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        image = cv2.equalizeHist(image)[..., None]

        if self._transforms is not None:
            image, coord = self._transforms(image, coord)

        return image, coord


if __name__ == "__main__":
    ds = MagazineCropDataset(split="train", transforms=build_scanned_transforms())

    img, mask = ds[1]

    print(img.shape, mask.shape)
    print(mask.unique())
