# dev/scripts/validate_dataset.py

import os
import numpy as np
import torch
from detection.dataset import RCCPatchDataset
from torch.utils.data import DataLoader
import argparse

def main(processed_dir, batch_size=8, augment=False, split='train', max_batches=2):
    # QC on the patch dataset
    print(f"Loading RCCPatchDataset from: {processed_dir}")
    dataset = RCCPatchDataset(processed_dir, split=split, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Number of samples: {len(dataset)}")
    print(f"Sample label counts: {np.bincount(dataset.labels)}")
    print("Sample meta example:", dataset.metas[0])
    
    for i, (imgs, labels, metas) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print("  Images:", imgs.shape, imgs.dtype)
        print("  Labels:", labels.shape, labels.dtype, "Labels:", labels.numpy())
        print("  Meta:", metas[0])
        if augment:
            print("  (Augmentation enabled, check visually or use assert statements for randomness)")
        if i >= max_batches - 1:
            break

    # Optional: Visualize a few patches to ensure augmentations work as intended
    try:
        import matplotlib.pyplot as plt
        imgs, labels, metas = next(iter(loader))
        for i in range(min(4, imgs.shape[0])):
            plt.imshow(imgs[i,0].numpy(), cmap='gray')
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')
            plt.show()
    except ImportError:
        print("matplotlib not installed, skipping visualization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='dev/data/processed/kits23/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_batches', type=int, default=2)
    args = parser.parse_args()
    main(args.processed_dir, args.batch_size, args.augment, args.split, args.max_batches)
