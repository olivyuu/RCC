# dev/detection/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

class RCCPatchDataset(Dataset):
    """
    Dataset for patch-based training for RCC kidney lesion detection (tumor or cyst).
    Expects .npz files with arrays: patches, masks, meta (1 file per case).
    Returns (image, label, meta_dict) per patch.
    Label: 1 = tumor or cyst present, 0 = background/empty/kidney.
    """

    def __init__(self, data_dir, split='train', split_seed=42, split_frac=0.8,
                 augment=False, patch_file_suffix='_patches.npz', transform=None):
        """
        Args:
            data_dir: folder with *_patches.npz files
            split: "train" or "val"
            split_seed: reproducibility for split
            split_frac: train/val split fraction
            augment: whether to use augmentations (overrides transform if True)
            transform: torchvision transforms (applied to images only)
        """
        self.data_dir = data_dir
        self.patch_files = sorted([f for f in os.listdir(data_dir) if f.endswith(patch_file_suffix)])
        random.seed(split_seed)
        # Deterministic split by file
        n = len(self.patch_files)
        idxs = list(range(n))
        random.shuffle(idxs)
        split_point = int(n * split_frac)
        if split == 'train':
            chosen = [self.patch_files[i] for i in idxs[:split_point]]
        else:
            chosen = [self.patch_files[i] for i in idxs[split_point:]]
        self.selected_files = chosen

        # Load all patches and labels into memory for speed (can switch to lazy if OOM)
        self.images = []
        self.labels = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches']  # (N, 224, 224, 1)
            masks = data['masks']
            meta = data['meta']
            # Label: 1 if patch contains tumor OR cyst; 0 otherwise
            labels = np.array([
                ((mask[...,0] == 2).any() or (mask[...,0] == 3).any())
                for mask in masks
            ]).astype(np.int64)
            self.images.extend([patch[...,0] for patch in patches])
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images)  # (N, 224, 224)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        # Always convert to tensor first, then augment if requested
        if self.augment:
            t_list = [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ]
        else:
            t_list = [transforms.ToTensor()]
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape [224, 224]
        label = self.labels[idx]
        meta = self.metas[idx]
        img = img.astype(np.float32)
        img = self.transform(img)
        return img, label, meta
