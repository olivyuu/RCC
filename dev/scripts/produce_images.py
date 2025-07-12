import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
QC_DIR = "dev/data/qc/three_plane_examples/"
os.makedirs(QC_DIR, exist_ok=True)

def save_matched_pair(img2d, mask2d, fname_prefix, include_tumor):
    # Raw image
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img2d, cmap='gray')
    ax.axis('off')
    plt.savefig(f"{fname_prefix}_raw.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved raw image: {fname_prefix}_raw.png")

    # Overlay
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img2d, cmap='gray')
    mask_drawn = False
    if np.any(mask2d == 1):
        ax.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
        mask_drawn = True
    if include_tumor and np.any(mask2d == 2):
        ax.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
        mask_drawn = True
    ax.axis('off')
    plt.savefig(f"{fname_prefix}_overlay.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if mask_drawn:
        print(f"Saved overlay image: {fname_prefix}_overlay.png")
    else:
        print(f"Saved overlay image (no mask present in this slice): {fname_prefix}_overlay.png")

def find_slice(mask, orientation, require_tumor_and_kidney=False, require_kidney_only=False):
    # orientation: 'axial', 'sagittal', or 'coronal'
    # Returns: slice idx (int) or None
    axis = {'axial':2, 'sagittal':0, 'coronal':1}[orientation]
    n_slices = mask.shape[axis]
    for sl in range(n_slices):
        mask2d = mask[:, :, sl] if orientation=='axial' else mask[sl, :, :] if orientation=='sagittal' else mask[:, sl, :]
        if require_tumor_and_kidney:
            if np.any(mask2d == 1) and np.any(mask2d == 2):
                return sl
        if require_kidney_only:
            if np.any(mask2d == 1) and not np.any(mask2d == 2) and not np.any(mask2d == 3):
                return sl
    return None

# ----------- Find case and slices for kidney+tumor -----------
case1_found = False
print("Searching for case with kidney+tumor slices...")
for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    data = np.load(os.path.join(PROCESSED_DIR, f))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
    found_all = True
    slices = {}
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = find_slice(mask, orient, require_tumor_and_kidney=True)
        if sl is None:
            found_all = False
            break
        slices[orient] = sl
    if found_all:
        print(f"\n[INFO] Selected {case_id} for 'kidney_and_tumor' example")
        for orient in ['axial', 'sagittal', 'coronal']:
            sl = slices[orient]
            img2d = img[:, :, sl] if orient == 'axial' else img[sl, :, :] if orient == 'sagittal' else img[:, sl, :]
            mask2d = mask[:, :, sl] if orient == 'axial' else mask[sl, :, :] if orient == 'sagittal' else mask[:, sl, :]
            fname_prefix = os.path.join(QC_DIR, f"{case_id}_kidney_and_tumor_{orient}")
            save_matched_pair(img2d, mask2d, fname_prefix, include_tumor=True)
        case1_found = True
        break

# ----------- Find kidney-only slices (may be in same or any other case) -----------
case2_found = False
print("Searching for slices with kidney only (no tumor)...")
for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    data = np.load(os.path.join(PROCESSED_DIR, f))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
    found = 0
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = find_slice(mask, orient, require_kidney_only=True)
        if sl is not None:
            img2d = img[:, :, sl] if orient == 'axial' else img[sl, :, :] if orient == 'sagittal' else img[:, sl, :]
            mask2d = mask[:, :, sl] if orient == 'axial' else mask[sl, :, :] if orient == 'sagittal' else mask[:, sl, :]
            fname_prefix = os.path.join(QC_DIR, f"{case_id}_kidney_only_{orient}")
            save_matched_pair(img2d, mask2d, fname_prefix, include_tumor=False)
            found += 1
        if found == 3:
            print(f"[INFO] Selected {case_id} for kidney-only slices in all orientations.")
            case2_found = True
            break
    if case2_found:
        break

print("Done.")
