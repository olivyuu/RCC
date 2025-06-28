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

def extract_patches(img, mask, patch_size, background_ratio=1, true_bg_per_case=5):
    # Lesion = (mask==2 or mask==3), Kidney-background = (mask==1 only), True background = (mask==0)
    slices = mask.shape[2]
    patch_list, mask_list, meta_list = [], [], []

    # --- 1. Lesion patches (tumor or cyst) ---
    for sl in range(slices):
        lesion_mask = np.logical_or(mask[:,:,sl] == 2, mask[:,:,sl] == 3)
        if not lesion_mask.any():
            continue
        # Use the whole slice as patch (fits patch size)
        img_patch = img[:,:,sl:sl+1]
        mask_patch = mask[:,:,sl:sl+1]
        patch_list.append(img_patch)
        mask_list.append(mask_patch)
        meta_list.append({"slice": int(sl), "type": "lesion"})

        # --- 2. Kidney-background patches ---
        bg_mask = (mask[:,:,sl] == 1)
        bg_idxs = np.argwhere(bg_mask)
        if len(bg_idxs) > 0 and background_ratio > 0:
            select_n = min(background_ratio, len(bg_idxs))
            chosen = bg_idxs[np.random.choice(len(bg_idxs), select_n, replace=False)]
            for i, j in chosen:
                if i + patch_size[0] > img.shape[0] or j + patch_size[1] > img.shape[1]:
                    continue
                img_patch_bg = img[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
                mask_patch_bg = mask[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
                if img_patch_bg.shape == patch_size and mask_patch_bg.shape == patch_size:
                    patch_list.append(img_patch_bg)
                    mask_list.append(mask_patch_bg)
                    meta_list.append({"slice": int(sl), "type": "background", "i": int(i), "j": int(j)})

    # --- 3. True-background patches (anywhere with mask==0) ---
    all_true_bg = np.argwhere(mask == 0)
    if len(all_true_bg) > 0:
        chosen_true_bg = all_true_bg[np.random.choice(len(all_true_bg), min(true_bg_per_case, len(all_true_bg)), replace=False)]
        for i, j, sl in chosen_true_bg:
            if i + patch_size[0] > img.shape[0] or j + patch_size[1] > img.shape[1]:
                continue
            img_patch_bg = img[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            mask_patch_bg = mask[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            if img_patch_bg.shape == patch_size and mask_patch_bg.shape == patch_size:
                patch_list.append(img_patch_bg)
                mask_list.append(mask_patch_bg)
                meta_list.append({"slice": int(sl), "type": "true_background", "i": int(i), "j": int(j)})
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
