import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RCCPatchDataset(Dataset):
    def __init__(self, data_dir, split='train', split_seed=42, split_frac=0.8,
                 augment=False, patch_file_suffix='_patches.npz', transform=None):
        self.data_dir = data_dir
        self.patch_files = sorted([f for f in os.listdir(data_dir) if f.endswith(patch_file_suffix)])
        np.random.seed(split_seed)
        # Deterministic split by file
        n = len(self.patch_files)
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        split_point = int(n * split_frac)
        if split == 'train':
            chosen = [self.patch_files[i] for i in idxs[:split_point]]
        else:
            chosen = [self.patch_files[i] for i in idxs[split_point:]]
        self.selected_files = chosen

        # Load all patches and labels into memory for speed
        self.images = []
        self.labels = []
        self.masks = []
        self.metas = []
        for pf in self.selected_files:
            data = np.load(os.path.join(data_dir, pf), allow_pickle=True)
            patches = data['patches']    # (N, 224, 224, 1)
            masks = data['masks']        # (N, 224, 224, 1)
            meta = data['meta']
            # Label: lesion (tumor/cyst) = 1 if any pixel==2 or 3 in mask
            labels = np.array([(((mask[...,0] == 2) | (mask[...,0] == 3)).any()) for mask in masks]).astype(np.int64)
            self.images.extend([patch[...,0] for patch in patches])
            self.masks.extend([mask[...,0] for mask in masks])
            self.labels.extend(labels)
            self.metas.extend(meta)
        self.images = np.stack(self.images)   # (N, 224, 224)
        self.masks = np.stack(self.masks)     # (N, 224, 224)
        self.labels = np.array(self.labels)
        self.metas = np.array(self.metas, dtype=object)

        self.augment = augment
        self.transform = transform if transform else self.default_transform()
        self.mask_transform = transforms.ToTensor()  # For masks

    def default_transform(self):
        t_list = []
        if self.augment:
            t_list = [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Only works for images, not masks
                transforms.ToTensor(),
            ]
        else:
            t_list = [transforms.ToTensor()]
        return transforms.Compose(t_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)    # (224, 224)
        mask = self.masks[idx].astype(np.uint8)      # (224, 224)
        label = self.labels[idx]
        meta = self.metas[idx]
        img = np.expand_dims(img, axis=0)            # (1, 224, 224) for ToPILImage

        # Augment both image and mask (but only flips/rotations, not ColorJitter!)
        if self.augment:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            img_torch = self.transform(img)
            torch.manual_seed(seed)
            mask = np.expand_dims(mask, axis=0)      # (1, 224, 224)
            mask_torch = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
            ])(mask)
            mask_torch = mask_torch[0]   # remove channel dim
        else:
            img_torch = self.transform(img)
            mask_torch = torch.from_numpy(mask).float()

        return img_torch, label, mask_torch, meta
