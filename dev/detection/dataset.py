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
    Label: 1 = lesion-present, 0 = background.
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

        # Load all patches, labels, masks, meta into memory for speed
        self.images = []
        self.labels = []
        self.masks = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches'] # (N, 224, 224, 1)
            masks = data['masks']     # (N, 224, 224, 1)
            meta = data['meta']
            # Label: lesion if any pixel in mask==2 or 3 (tumor/cyst)
            labels = np.array([np.any((mask[...,0] == 2) | (mask[...,0] == 3)) for mask in masks]).astype(np.int64)
            self.images.extend([patch[...,0] for patch in patches])
            self.labels.extend(labels)
            self.masks.extend([mask[...,0] for mask in masks])
            self.metas.extend(meta)
        self.images = np.stack(self.images) # (N, 224, 224)
        self.labels = np.array(self.labels)
        self.masks = np.stack(self.masks)
        self.metas = np.array(self.metas)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()

    def default_transform(self):
        t_list = [transforms.ToTensor()]
        if self.augment:
            t_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ]
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]         # [224, 224]
        label = self.labels[idx]       # int
        mask = self.masks[idx]         # [224, 224]
        meta = self.metas[idx]         # dict

        # To [1, H, W] and float32 for transforms
        img = np.expand_dims(img, axis=0).astype(np.float32)  # [1, 224, 224]
        mask = mask.astype(np.uint8)

        # Only apply transform to image
        img_torch = self.transform(img)   # [1, 224, 224]
        # Do NOT transform mask, just convert to torch
        mask_torch = torch.from_numpy(mask).long()    # [224, 224]

        return img_torch, label, mask_torch, meta
