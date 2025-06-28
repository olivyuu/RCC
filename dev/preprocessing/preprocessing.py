"""
Preprocessing pipeline for patch-based kidney tumor/cyst detection on KiTS23 dataset.
- Loads cases from RCC/kits23/dataset/
- Outputs processed cases and patch data to RCC/dev/data/processed/kits23/
- For each case, extracts:
    * Lesion patches (tumor or cyst, mask==2 or 3)
    * Kidney-background patches (kidney only, mask==1)
    * True-background patches (non-kidney, non-lesion, mask==0)
"""

import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from skimage.transform import resize

# ========== CONFIGURATION ========== #
RAW_DATA_DIR = "kits23/dataset/"
PROCESSED_DATA_DIR = "dev/data/processed/kits23/"
TARGET_SHAPE = (224, 224, 64)  # (H, W, Slices)
PATCH_SIZE = (224, 224, 1)     # Single-slice patches, [H, W, 1]
BACKGROUND_RATIO = 1           # Kidney-background:lesion ratio per slice
TRUE_BG_PER_CASE = 5           # Number of "true background" (non-kidney, non-lesion) patches per case
SEED = 42

np.random.seed(SEED)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize_img(img):
    img = np.clip(img, -200, 300)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def resize_volume(vol, target_shape, order=1):
    return resize(vol, target_shape, order=order, mode='edge', anti_aliasing=(order != 0))

def extract_patches(img, mask, patch_size, background_ratio=1, true_bg_per_case=10):
    slices = mask.shape[2]
    patch_list, mask_list, meta_list = [], [], []

    lesion_indices = []
    kidney_bg_indices = []
    true_bg_indices = []

    for sl in range(slices):
        # 1. Lesion: tumor or cyst
        lesion_mask = np.logical_or(mask[:,:,sl] == 2, mask[:,:,sl] == 3)
        if lesion_mask.any():
            patch_list.append(img[:,:,sl:sl+1])
            mask_list.append(mask[:,:,sl:sl+1])
            meta_list.append({"slice": int(sl), "type": "lesion"})
            lesion_indices.append(sl)
        # 2. Kidney-background (any slice with kidney, but not lesion in center)
        elif (mask[:,:,sl] == 1).any():
            kidney_bg_indices.append(sl)
        # 3. True-background: slices with no kidney, no lesion
        elif (mask[:,:,sl] == 0).all():
            true_bg_indices.append(sl)

    # Sample kidney-background patches from kidney_bg_indices
    n_kidney_bg = background_ratio * max(1, len(lesion_indices))  # for balance
    chosen_kidney = np.random.choice(kidney_bg_indices, min(n_kidney_bg, len(kidney_bg_indices)), replace=False)
    for sl in chosen_kidney:
        patch_list.append(img[:,:,sl:sl+1])
        mask_list.append(mask[:,:,sl:sl+1])
        meta_list.append({"slice": int(sl), "type": "background"})

    # Sample true-background patches (no kidney, no lesion)
    n_true_bg = true_bg_per_case
    if len(true_bg_indices) > 0:
        chosen_true = np.random.choice(true_bg_indices, min(n_true_bg, len(true_bg_indices)), replace=False)
        for sl in chosen_true:
            patch_list.append(img[:,:,sl:sl+1])
            mask_list.append(mask[:,:,sl:sl+1])
            meta_list.append({"slice": int(sl), "type": "true_background"})

    return patch_list, mask_list, meta_list


def process_case(case_dir, case_id):
    img_path = glob(os.path.join(case_dir, "*imaging.nii.gz"))[0]
    mask_path = glob(os.path.join(case_dir, "*segmentation.nii.gz"))[0]
    img = load_nifti(img_path)
    mask = load_nifti(mask_path)
    img_resized = resize_volume(img, TARGET_SHAPE)
    mask_resized = resize_volume(mask, TARGET_SHAPE, order=0)
    img_resized = normalize_img(img_resized)

    patches, masks, metas = extract_patches(
        img_resized, mask_resized, PATCH_SIZE, background_ratio=BACKGROUND_RATIO, true_bg_per_case=TRUE_BG_PER_CASE
    )

    np.savez_compressed(os.path.join(PROCESSED_DATA_DIR, f"{case_id}_vol.npz"),
                        image=img_resized, mask=mask_resized)
    np.savez_compressed(os.path.join(PROCESSED_DATA_DIR, f"{case_id}_patches.npz"),
                        patches=np.array(patches), masks=np.array(masks), meta=np.array(metas, dtype=object))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=RAW_DATA_DIR, help="Path to raw KiTS23 data")
    parser.add_argument("--max_cases", type=int, default=None, help="Number of cases to process")
    args = parser.parse_args()
    case_dirs = sorted(glob(os.path.join(args.raw_dir, "case_*")))
    if args.max_cases is not None:
        case_dirs = case_dirs[:args.max_cases]
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_id = os.path.basename(case_dir)
        try:
            process_case(case_dir, case_id)
        except Exception as e:
            print(f"Failed {case_id}: {e}")

if __name__ == "__main__":
    main()
