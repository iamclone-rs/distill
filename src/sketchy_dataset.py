"""Datasets for the fixed CoPrompt training and zero-shot SBIR evaluation."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.data_config import GENERALIZED_CLASSES, UNSEEN_CLASSES, VISUALIZE_CLASSES


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augmented_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def normal_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def _class_names(root: Path):
    sketch_root = root / "sketch"
    photo_root = root / "photo"
    if not sketch_root.is_dir() or not photo_root.is_dir():
        raise FileNotFoundError(f"Expected sketch/ and photo/ under '{root}'.")
    return sorted(
        path.name
        for path in sketch_root.iterdir()
        if path.is_dir() and (photo_root / path.name).is_dir()
    )


def _images(directory: Path):
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in '{directory}'.")
    return images


def _load_rgb(path: Path, image_size: int):
    with Image.open(path) as image:
        return ImageOps.pad(
            image.convert("RGB"),
            size=(image_size, image_size),
        )


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset, image_size=224):
        self.root = Path(root)
        self.dataset = dataset
        self.image_size = image_size
        self.clean_transform = normal_transform(image_size)
        self.augmented_transform = augmented_transform(image_size)

        unseen = set(UNSEEN_CLASSES[dataset])
        self.categories = [name for name in _class_names(self.root) if name not in unseen]
        if len(self.categories) < 2:
            raise RuntimeError("Training requires at least two seen classes.")
        self.class_to_label = {name: index for index, name in enumerate(self.categories)}
        self.sketches = []
        self.photos = {}
        for category in self.categories:
            self.sketches.extend(_images(self.root / "sketch" / category))
            self.photos[category] = _images(self.root / "photo" / category)

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, index):
        sketch_path = self.sketches[index]
        category = sketch_path.parent.name
        negative_categories = [name for name in self.categories if name != category]
        photo_path = np.random.choice(self.photos[category])
        negative_category = np.random.choice(negative_categories)
        negative_path = np.random.choice(self.photos[negative_category])

        sketch = _load_rgb(sketch_path, self.image_size)
        photo = _load_rgb(Path(photo_path), self.image_size)
        negative = _load_rgb(Path(negative_path), self.image_size)
        return (
            self.clean_transform(photo),
            self.clean_transform(sketch),
            self.augmented_transform(photo),
            self.augmented_transform(sketch),
            self.clean_transform(negative),
            self.class_to_label[category],
        )


class ValidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        dataset,
        modality="photo",
        *,
        generalized=False,
        visualize=False,
        image_size=224,
    ):
        self.root = Path(root)
        self.dataset = dataset
        self.modality = modality
        self.image_size = image_size
        self.transform = normal_transform(image_size)

        if visualize:
            if dataset not in VISUALIZE_CLASSES:
                raise ValueError(f"Visualization classes are not configured for '{dataset}'.")
            query_classes = list(VISUALIZE_CLASSES[dataset])
        else:
            query_classes = list(UNSEEN_CLASSES[dataset])
        self.categories = list(query_classes)
        if generalized and modality == "photo":
            if dataset not in GENERALIZED_CLASSES:
                raise ValueError(f"GZS gallery classes are not configured for '{dataset}'.")
            self.categories.extend(
                sorted(set(GENERALIZED_CLASSES[dataset]) - set(self.categories))
            )
        self.class_to_label = {name: index for index, name in enumerate(self.categories)}

        self.samples = []
        classes_to_load = self.categories if modality == "photo" else query_classes
        for category in classes_to_load:
            self.samples.extend(
                (path, self.class_to_label[category])
                for path in _images(self.root / modality / category)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = _load_rgb(path, self.image_size)
        return self.transform(image), label
