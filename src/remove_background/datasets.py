from torchvision import datasets as tv_datasets


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
