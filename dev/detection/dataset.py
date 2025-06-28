# dev/detection/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

class RCCPatchDataset(Dataset):
    """
    Dataset for patch-based training for RCC tumor detection.
    Expects .npz files with arrays: patches, masks, meta (1 file per case).
    Returns (image, label, mask, meta_dict) per patch.
    Label: 1 = lesion-present (tumor/cyst), 0 = background (kidney or true background).
    """

    def __init__(
        self, 
        data_dir, 
        split='train', 
        split_seed=42, 
        split_frac=0.8,
        augment=False, 
        patch_file_suffix='_patches.npz', 
        transform=None
    ):
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

        # Load all patches, masks, labels, and meta into memory (can be refactored for lazy loading if OOM)
        self.images = []
        self.masks = []
        self.labels = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches']  # (N, 224, 224, 1)
            masks = data['masks']      # (N, 224, 224, 1)
            meta = data['meta']
            # Label: lesion if any pixel in mask==2 (tumor) or ==3 (cyst)
            labels = np.array([(np.logical_or(mask[..., 0] == 2, mask[..., 0] == 3)).any() for mask in masks]).astype(np.int64)
            self.images.extend([patch[..., 0] for patch in patches])   # [H, W]
            self.masks.extend([mask[..., 0] for mask in masks])        # [H, W] for each mask
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images)   # (N, 224, 224)
        self.masks = np.stack(self.masks)     # (N, 224, 224)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        t_list = []
        if self.augment:
            t_list = [
                transforms.ToPILImage(),  # Accepts [C,H,W] uint8 or float32 in 0-1
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30, fill=0),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # ±5% shift
                    scale=(0.95, 1.05),
                    shear=5,
                    fill=0
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  # Returns float tensor [C,H,W] in [0,1]
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),  # Add slight Gaussian noise
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Ensure values in [0,1]
            ]
        else:
            t_list = [transforms.ToTensor()]  # Just convert to tensor
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]     # [H, W]
        mask = self.masks[idx]     # [H, W] (uint8, values: 0-bg, 1-kidney, 2-tumor, 3-cyst)
        label = self.labels[idx]
        meta = self.metas[idx]
        # Convert to PIL expects [H,W] or [C,H,W], uint8 or float32 in [0,1]
        img = img.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
            # Mask is not augmented (for segmentation, sync transforms would be required)
        mask = torch.from_numpy(mask.astype(np.uint8))
        return img, label, mask, meta
