import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

class RCCPatchDataset(Dataset):
    """
    Dataset for patch-based training for RCC lesion detection.
    Handles lesion, background-kidney, and background-other patches.
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

        # Load patches/labels
        self.images = []
        self.labels = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches'] # (N, 224, 224, 1)
            meta = data['meta']
            # Determine label by patch type
            for img, m in zip(patches, meta):
                t = m['type']
                if t == 'lesion':
                    self.labels.append(1)
                elif t in ['background-kidney', 'background-other']:
                    self.labels.append(0)
                else:
                    raise ValueError(f"Unknown patch type in meta: {t}")
                self.images.append(img[..., 0])
                self.metas.append(m)
        self.images = np.stack(self.images)
        self.labels = np.array(self.labels)
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
                transforms.ToTensor(),
            ]
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        meta = self.metas[idx]
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = self.transform(img)
        return img, label, meta
