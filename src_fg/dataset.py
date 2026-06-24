import os
import glob
import numpy as np
import torch
import math 
from torch.nn import functional as F

from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES
        
def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(
            0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return dataset_transforms


class SketchyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        unseen_classes = UNSEEN_CLASSES["sketchy_2"]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.transform = normal_transform()
        self.aumentation = aumented_transform()

        if self.mode == "train":
            self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        else:
            self.all_categories = sorted(list(set(unseen_classes)))

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.photo_id_map = {}
        
        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))
            photo_paths = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
            self.all_photos_path[category] = photo_paths
            for p in photo_paths:
                key = os.path.normcase(os.path.normpath(p))
                if key not in self.photo_id_map:
                    self.photo_id_map[key] = len(self.photo_id_map)
                    
    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]

        pos_sample = sk_path.split('/')[-1].split('-')[:-1][0]
        pos_path = glob.glob(os.path.join(self.args.root, 'photo', category, pos_sample + '.*'))
        if len(pos_path) == 0:
            print(sk_path)
            return None

        pos_path = pos_path[0]
        photo_category = self.all_photos_path[category].copy()
        photo_category = [p for p in photo_category if p != pos_path]

        neg_path = np.random.choice(photo_category)

        sk_data = Image.fromarray(np.array(Image.open(sk_path).convert('RGB')))
        img_data = Image.fromarray(np.array(Image.open(pos_path).convert('RGB')))
        neg_data = Image.fromarray(np.array(Image.open(neg_path).convert('RGB')))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.mode == "train":
            sk_aug_tensor = self.aumentation(sk_data)
            img_aug_tensor = self.aumentation(img_data)
            return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.all_categories.index(category)

        else:
            return sk_tensor, sk_path, img_tensor, pos_sample, self.all_categories.index(category)