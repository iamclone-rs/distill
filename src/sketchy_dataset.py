import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ]
    return transforms.Compose(transform_list)

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    return dataset_transforms


def _instance_id_from_path(path):
    """Return the photo instance id used by Sketchy fine-grained matching."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("-")
    if len(parts) > 1:
        return "-".join(parts[:-1])
    return stem


def _positive_photo_path(root, category, sketch_path):
    instance_id = _instance_id_from_path(sketch_path)
    candidates = sorted(
        glob.glob(os.path.join(root, 'photo', category, instance_id + '.*'))
    )
    if len(candidates) == 0:
        return None
    return candidates[0]


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.transform1 = normal_transform()
        self.transform2 = aumented_transform()
        
        unseen_classes = UNSEEN_CLASSES[self.args.dataset]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        
        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketch_paths = glob.glob(os.path.join(self.args.root, 'sketch', category, '*'))
            photo_paths = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
            
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]

        sk_path  = filepath
        if getattr(self.args, "fine_grained", False):
            instance_id = _instance_id_from_path(sk_path)
            img_path = _positive_photo_path(self.args.root, category, sk_path)
            if img_path is None:
                raise FileNotFoundError(
                    f"No fine-grained positive photo for sketch '{sk_path}' "
                    f"(expected photo/{category}/{instance_id}.*)"
                )
            same_category_negatives = [
                p for p in self.all_photos_path[category]
                if os.path.normcase(os.path.normpath(p))
                != os.path.normcase(os.path.normpath(img_path))
            ]
            if len(same_category_negatives) == 0:
                raise RuntimeError(
                    f"Need at least two photos in category '{category}' for fine-grained triplet."
                )
            neg_path = np.random.choice(same_category_negatives)
        else:
            neg_classes = self.all_categories.copy()
            neg_classes.remove(category)
            img_path = np.random.choice(self.all_photos_path[category])
            neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        sk_tensor  = self.transform1(sk_data)
        img_tensor = self.transform1(img_data)
        neg_tensor = self.transform1(neg_data)
        
        sk_aug_tensor = self.transform2(sk_data)
        img_aug_tensor = self.transform2(img_data)
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.all_categories.index(category)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        self.unseen_classes = UNSEEN_CLASSES[self.args.dataset]
            
        unseen_paths = []
        for category in self.unseen_classes:
            if getattr(self.args, "fine_grained", False) and self.mode == 'photo':
                matched_photo_paths = []
                sketch_paths = sorted(
                    glob.glob(os.path.join(self.args.root, 'sketch', category, '*'))
                )
                for sketch_path in sketch_paths:
                    photo_path = _positive_photo_path(self.args.root, category, sketch_path)
                    if photo_path is not None:
                        matched_photo_paths.append(photo_path)
                unseen_paths.extend(sorted(set(matched_photo_paths)))
            elif self.mode == 'photo':
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
            else:
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))

        self.paths = sorted(unseen_paths)

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        label = self.unseen_classes.index(category)
        if getattr(self.args, "fine_grained", False):
            return image_tensor, label, category, _instance_id_from_path(filepath)
        return image_tensor, label
    
    def __len__(self):
        return len(self.paths)
