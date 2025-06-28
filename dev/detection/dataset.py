# dev/detection/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
from torchvision import transforms

class RCCPatchDataset(Dataset):
    """
    Dataset for patch-based training for RCC tumor detection.
    Expects .npz files with arrays: patches, masks, meta (1 file per case).
    Returns (image, label, meta_dict) per patch.
    Label: 1 = lesion-present (tumor/cyst), 0 = background (including kidney and "true background" patches).
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
        """
        Args:
            data_dir: folder with *_patches.npz files
            split: "train" or "val"
            split_seed: reproducibility for split
            split_frac: train/val split fraction
            augment: whether to use augmentations (overrides transform if True)
            transform: torchvision transforms (applied to torch.Tensor images)
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
            # Label: lesion if any pixel in mask==2 (tumor) or ==3 (cyst)
            labels = np.array([(np.logical_or(mask[..., 0] == 2, mask[..., 0] == 3)).any() for mask in masks]).astype(np.int64)
            self.images.extend([patch[..., 0] for patch in patches])  # [H, W]
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images)  # (N, 224, 224)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        t_list = []
        if self.augment:
            t_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
            ]
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # [H, W]
        label = self.labels[idx]
        meta = self.metas[idx]
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        img = torch.from_numpy(img.astype(np.float32))  # ensure tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, label, meta
