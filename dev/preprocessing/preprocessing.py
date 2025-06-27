"""
Preprocessing pipeline for patch-based kidney tumor/cyst detection on KiTS23 dataset.
- Loads cases from RCC/kits23/dataset/
- Outputs processed cases and patch data to RCC/dev/data/processed/kits23/
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
TARGET_SHAPE = (224, 224, 64)  # Downsampled (H, W, Slices)
PATCH_SIZE = (224, 224, 1)     # Patch = 1 slice (for 2D CNN, but stored in 3D format)
BACKGROUND_RATIO = 1           # Background:kidney region ratio per tumor/cyst patch
TRUE_BG_RATIO = 0.2            # Ratio of "true" background (no kidney) patches to total lesion patches
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

def extract_patches(img, mask, patch_size, suspicious_labels=(2, 3), background_ratio=1, true_bg_ratio=0.2):
    # suspicious_labels: (tumor=2, cyst=3)
    H, W, S = img.shape
    patch_list, mask_list, meta_list = [], [], []

    # (1) Lesion (tumor or cyst) patches
    lesion_voxels = np.argwhere(np.isin(mask, suspicious_labels))
    slices = np.unique(lesion_voxels[:,2]) if len(lesion_voxels) > 0 else []
    for sl in slices:
        img_patch = img[:,:,sl:sl+1]
        mask_patch = mask[:,:,sl:sl+1]
        # meta: "suspicious" (label=1)
        patch_list.append(img_patch)
        mask_list.append(mask_patch)
        meta_list.append({"slice": int(sl), "type": "lesion"})

        # (2) Sample background patches in kidney region (mask==1, not suspicious)
        bg_mask = (mask[:,:,sl] == 1) & (~np.isin(mask[:,:,sl], suspicious_labels))
        bg_idxs = np.argwhere(bg_mask)
        if len(bg_idxs) == 0:
            continue
        select_n = min(background_ratio, len(bg_idxs))
        chosen = bg_idxs[np.random.choice(len(bg_idxs), select_n, replace=False)]
        for idx in chosen:
            i, j = idx
            if i + patch_size[0] > H or j + patch_size[1] > W:
                continue
            img_patch_bg = img[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            mask_patch_bg = mask[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            if img_patch_bg.shape == patch_size and mask_patch_bg.shape == patch_size:
                patch_list.append(img_patch_bg)
                mask_list.append(mask_patch_bg)
                meta_list.append({"slice": int(sl), "type": "kidney_background", "i": int(i), "j": int(j)})

    # (3) True background (no kidney at all) patches
    n_true_bg = int(len(patch_list) * true_bg_ratio)
    for _ in range(n_true_bg):
        tries = 0
        while tries < 20:
            sl = np.random.randint(0, S)
            i = np.random.randint(0, H - patch_size[0] + 1)
            j = np.random.randint(0, W - patch_size[1] + 1)
            mask_patch = mask[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            if np.all(mask_patch == 0):  # fully outside kidney region
                img_patch = img[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
                patch_list.append(img_patch)
                mask_list.append(mask_patch)
                meta_list.append({"slice": int(sl), "type": "true_background", "i": int(i), "j": int(j)})
                break
            tries += 1

    return patch_list, mask_list, meta_list

def process_case(case_dir, case_id):
    img_path = glob(os.path.join(case_dir, "*imaging.nii.gz"))[0]
    mask_path = glob(os.path.join(case_dir, "*segmentation.nii.gz"))[0]
    img = load_nifti(img_path)
    mask = load_nifti(mask_path)

    img_resized = resize_volume(img, TARGET_SHAPE)
    mask_resized = np.round(resize_volume(mask, TARGET_SHAPE, order=0)).astype(np.uint8)
    img_resized = normalize_img(img_resized)

    # Extract patches: both tumor and cyst are "suspicious" for detection
    patches, masks, metas = extract_patches(
        img_resized, mask_resized, PATCH_SIZE,
        suspicious_labels=(2, 3),  # tumor=2, cyst=3
        background_ratio=BACKGROUND_RATIO,
        true_bg_ratio=TRUE_BG_RATIO
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
