from torchvision import transforms as trans


def build_transforms():
    tr_fn = trans.Compose(
        [
            trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            trans.RandomHorizontalFlip(),
            trans.RandomResizedCrop(size=(224, 224)),
            trans.ToTensor(),
            trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return tr_fn


def build_valid_transform():
    tr_fn = trans.Compose(
        [
            trans.Resize(size=(256, 256)),
            trans.CenterCrop(size=(224, 224)),
            trans.ToTensor(),
            trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return tr_fn
