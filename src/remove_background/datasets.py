import os
import json

import cv2
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets

from .transforms import build_scanned_transforms
from ..utils.misc import polygon_to_mask


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

        self._path_to_root = os.path.join(
            os.getcwd(), "data", f"{split}_data", "scanned"
        )
        with open(
            os.path.join(self._path_to_root, "annotations.json"), "r"
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
        image = cv2.imread(os.path.join(self._path_to_root, annotation["imagePath"]))
        mask = polygon_to_mask(
            polygon=annotation["shapes"][0]["points"],
            height=annotation["imageHeight"],
            width=annotation["imageWidth"],
        )

        img, mask = self._transforms(image, mask)

        return img, mask


if __name__ == "__main__":
    ds = MagazineCropDataset(split="train", transforms=build_scanned_transforms())

    img, mask = ds[1]

    print(img.shape, mask.shape)
    print(mask.unique())
