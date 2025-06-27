# dev/scripts/validate_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from dev.detection.dataset import RCCPatchDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to preprocessed patches (.npz)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max_batches", type=int, default=2, help="Number of batches to preview")
    parser.add_argument("--augment", action="store_true", help="Enable augmentations")
    args = parser.parse_args()

    print(f"Loading RCCPatchDataset from {args.processed_dir} [split: {args.split}] ...")
    dataset = RCCPatchDataset(
        data_dir=args.processed_dir,
        split=args.split,
        augment=args.augment
    )
    print(f"Loaded {len(dataset)} samples.")
    print("Class counts:", np.bincount(dataset.labels))
    print("Example meta:", dataset.metas[0])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    os.makedirs("dev/data/qc/patches/", exist_ok=True)

    for i, (imgs, labels, metas) in enumerate(loader):
        print(f"\n--- Batch {i+1} ---")
        print("  Images:", imgs.shape, imgs.dtype)
        print("    min/max:", imgs.min().item(), imgs.max().item())
        print("  Labels:", labels.shape, labels.dtype, "->", labels.numpy())
        # Print first 3 meta dicts for sanity check
        if isinstance(metas, (list, np.ndarray)):
            print("  Meta example:", metas[:3])
        else:
            print("  Meta (type?):", type(metas))
        if args.augment:
            print("  [Augmentation enabled]")
        # Save a few sample patches for visual QC (only first batch)
        if i == 0:
            import matplotlib.pyplot as plt
            for j in range(min(4, imgs.shape[0])):
                out_path = f"dev/data/qc/patches/batch{i+1}_img{j+1}.png"
                plt.imsave(out_path, imgs[j, 0].cpu().numpy(), cmap='gray')
                print(f"  Saved patch: {out_path}")
        if i >= args.max_batches - 1:
            break

if __name__ == "__main__":
    main()
