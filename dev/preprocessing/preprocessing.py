"""
Preprocessing pipeline for patch-based kidney tumor detection on KiTS23 dataset.
- Loads cases from RCC/dev/data/raw/kits23/
- Outputs processed cases and patch data to RCC/dev/data/processed/
- Saves PNG previews for QC to RCC/dev/data/qc/
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from skimage.transform import resize

# ========== CONFIGURATION ==========
RAW_DATA_DIR = "dev/data/raw/kits23/"
PROCESSED_DATA_DIR = "dev/data/processed/kits23/"
QC_DIR = "dev/data/qc/kits23/"
TARGET_SHAPE = (224, 224, 64)  # Downsampled (H, W, Slices)
PATCH_SIZE = (224, 224, 1)     # Patch = 1 slice (for 2D CNN, but stored in 3D format)
PATCH_OVERLAP = 0              # No overlap for now
BACKGROUND_RATIO = 1           # Background:tumor patch ratio (modifiable)
SEED = 42

np.random.seed(SEED)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize_img(img):
    # Clip to [-200, 300] HU, then min-max to [0,1]
    img = np.clip(img, -200, 300)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def resize_volume(vol, target_shape, order=1):
    # Resize 3D volume to target_shape
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return resize(vol, target_shape, order=order, mode='edge', anti_aliasing=True)

def save_qc_png(volume, mask, outpath, mask_name="mask"):
    # Show a middle slice
    mid = volume.shape[2] // 2
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.imshow(volume[:,:,mid], cmap="gray")
    plt.title("Image Slice")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(volume[:,:,mid], cmap="gray")
    plt.imshow(mask[:,:,mid], alpha=0.4, cmap="Reds")
    plt.title(f"{mask_name} Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def extract_patches(img, mask, patch_size, tumor_label=2, background_ratio=1):
    # Tumor mask: label==2 in kits23
    tumor_voxels = np.argwhere(mask == tumor_label)
    slices = np.unique(tumor_voxels[:,2])
    patch_list = []
    mask_list = []
    meta_list = []

    for sl in slices:
        img_patch = img[:,:,sl:sl+1]
        mask_patch = mask[:,:,sl:sl+1]
        patch_list.append(img_patch)
        mask_list.append(mask_patch)
        meta_list.append({"slice": int(sl), "type": "tumor"})
        
        # Sample background patches
        bg_mask = (mask[:,:,sl] == 1) & (mask[:,:,sl] != tumor_label)
        bg_idxs = np.argwhere(bg_mask)
        if len(bg_idxs) == 0:
            continue
        chosen = bg_idxs[np.random.choice(len(bg_idxs), min(background_ratio, len(bg_idxs)), replace=False)]
        for idx in chosen:
            i, j = idx
            img_patch_bg = img[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            if img_patch_bg.shape != patch_size:
                continue
            mask_patch_bg = mask[i:i+patch_size[0], j:j+patch_size[1], sl:sl+1]
            patch_list.append(img_patch_bg)
            mask_list.append(mask_patch_bg)
            meta_list.append({"slice": int(sl), "type": "background", "i": int(i), "j": int(j)})
    return patch_list, mask_list, meta_list

def process_case(case_dir, case_id):
    # Load files
    img_path = glob(os.path.join(case_dir, "*imaging.nii.gz"))[0]
    mask_path = glob(os.path.join(case_dir, "*segmentation.nii.gz"))[0]
    img = load_nifti(img_path)
    mask = load_nifti(mask_path)

    # Resize
    img_resized = resize_volume(img, TARGET_SHAPE)
    mask_resized = resize_volume(mask, TARGET_SHAPE, order=0)
    img_resized = normalize_img(img_resized)

    # Save QC overlays (all masks)
    for mask_label, mask_name in zip([1, 2, 3], ["kidney", "tumor", "cyst"]):
        bin_mask = (mask_resized == mask_label).astype(float)
        qc_path = os.path.join(QC_DIR, f"{case_id}_{mask_name}.png")
        save_qc_png(img_resized, bin_mask, qc_path, mask_name=mask_name)

    # Extract patches (just tumor + kidney background for now)
    patches, masks, metas = extract_patches(img_resized, mask_resized, PATCH_SIZE, tumor_label=2, background_ratio=BACKGROUND_RATIO)

    # Save processed full volume and masks as .npz
    np.savez_compressed(os.path.join(PROCESSED_DATA_DIR, f"{case_id}_vol.npz"),
                        image=img_resized, mask=mask_resized)
    # Save patches as .npz
    np.savez_compressed(os.path.join(PROCESSED_DATA_DIR, f"{case_id}_patches.npz"),
                        patches=np.array(patches), masks=np.array(masks), meta=np.array(metas, dtype=object))

def main():
    case_dirs = sorted(glob(os.path.join(RAW_DATA_DIR, "case_*")))
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_id = os.path.basename(case_dir)
        try:
            process_case(case_dir, case_id)
        except Exception as e:
            print(f"Failed {case_id}: {e}")

if __name__ == "__main__":
    main()
