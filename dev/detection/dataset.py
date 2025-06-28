import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class RCCPatchDataset(Dataset):
    """
    Dataset for patch-based training and evaluation for RCC tumor/cyst detection.
    Returns (image, label, mask, meta) per patch.
    """
    def __init__(self, data_dir, split='train', split_seed=42, split_frac=0.8,
                 augment=False, patch_file_suffix='_patches.npz', transform=None):
        self.data_dir = data_dir
        self.patch_files = sorted([f for f in os.listdir(data_dir) if f.endswith(patch_file_suffix)])
        random.seed(split_seed)
        n = len(self.patch_files)
        idxs = list(range(n))
        random.shuffle(idxs)
        split_point = int(n * split_frac)
        if split == 'train':
            chosen = [self.patch_files[i] for i in idxs[:split_point]]
        else:
            chosen = [self.patch_files[i] for i in idxs[split_point:]]
        self.selected_files = chosen

        self.images = []
        self.labels = []
        self.masks = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches'] # (N, 224, 224, 1)
            masks = data['masks']     # (N, 224, 224, 1)
            meta = data['meta']
            labels = np.array([(mask[...,0] == 2).any() or (mask[...,0] == 3).any() for mask in masks]).astype(np.int64)
            self.images.extend([patch[...,0] for patch in patches])
            self.masks.extend([mask[...,0] for mask in masks])
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images) # (N, 224, 224)
        self.masks = np.stack(self.masks)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        # Use torchvision transforms for PIL Images
        t_list = []
        if self.augment:
            t_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        t_list.append(transforms.ToTensor())
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        meta = self.metas[idx]
        # Convert to PIL Image for transforms that expect PIL
        # Scale to [0,255] and cast to uint8
        img_uint8 = (img * 255).clip(0,255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')
        img_torch = self.transform(img_pil)  # [1, 224, 224]
        # For mask, just convert to torch tensor (no augmentation)
        mask_torch = torch.from_numpy(mask).long()
        return img_torch, label, mask_torch, meta
