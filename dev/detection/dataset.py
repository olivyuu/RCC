import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image

class RCCPatchDataset(Dataset):
    """
    Patch-based dataset for RCC detection.
    Returns: (image_tensor, label, mask_tensor, meta)
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
            labels = np.array([((mask[...,0] == 2) | (mask[...,0] == 3)).any() for mask in masks]).astype(np.int64)
            # label=1 if tumor or cyst, else 0
            self.images.extend([patch[...,0] for patch in patches])
            self.masks.extend([mask[...,0] for mask in masks])
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images) # (N, 224, 224)
        self.masks = np.stack(self.masks)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas)

        self.augment = augment
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.build_transform()

    def build_transform(self):
        t_list = []
        if self.augment:
            # Augmentations must work with PIL Images.
            t_list.extend([
                transforms.Lambda(lambda img: Image.fromarray((img * 255).astype(np.uint8).squeeze())),  # (224,224) to PIL
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),                       # Stronger rotation
                transforms.ColorJitter(brightness=0.2, contrast=0.2),# Stronger jitter
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),  # Random translation
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), # Random crop and resize
                transforms.ToTensor(),  # Converts back to torch tensor [1,224,224]
            ])
        else:
            t_list.extend([
                transforms.Lambda(lambda img: Image.fromarray((img * 255).astype(np.uint8).squeeze())),
                transforms.ToTensor(),
            ])
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]    # [224, 224], float32
        mask = self.masks[idx]    # [224, 224], int
        label = self.labels[idx]  # int
        meta = self.metas[idx]    # dict or str

        # --- Augment image and mask together if augmenting ---
        if self.augment:
            seed = np.random.randint(0, 1e6)
            random.seed(seed)
            torch.manual_seed(seed)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            mask_pil = Image.fromarray((mask).astype(np.uint8))
            # Must apply spatial transforms identically:
            # Do spatial transforms on both, then apply ColorJitter only to image
            spatial = transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
            ])
            img_pil = spatial(img_pil)
            mask_pil = spatial(mask_pil)
            # ColorJitter only on image
            img_pil = transforms.ColorJitter(brightness=0.1, contrast=0.1)(img_pil)
            img_torch = transforms.ToTensor()(img_pil)
            mask_torch = torch.from_numpy(np.array(mask_pil)).long()
        else:
            img_torch = self.transform(img)
            mask_torch = torch.from_numpy(mask).long()

        return img_torch, label, mask_torch, meta
