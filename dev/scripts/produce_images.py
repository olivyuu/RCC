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

def best_slice(mask, orientation, kidney_and_tumor=False, kidney_only=False):
    axis = {'axial':2, 'sagittal':0, 'coronal':1}[orientation]
    n_slices = mask.shape[axis]
    best_idx = None
    best_val = -1
    for sl in range(n_slices):
        mask2d = mask[:, :, sl] if orientation=='axial' else mask[sl, :, :] if orientation=='sagittal' else mask[:, sl, :]
        if kidney_and_tumor:
            kidney = (mask2d == 1)
            tumor = (mask2d == 2)
            if kidney.any() and tumor.any():
                val = kidney.sum() + tumor.sum()
                if val > best_val:
                    best_val = val
                    best_idx = sl
        elif kidney_only:
            kidney = (mask2d == 1)
            tumor = (mask2d == 2)
            cyst  = (mask2d == 3)
            if kidney.any() and not tumor.any() and not cyst.any():
                val = kidney.sum()
                if val > best_val:
                    best_val = val
                    best_idx = sl
    return best_idx

# ----------- Find case and slices for kidney+tumor -----------
case1_found = False
case1_id = None
case1_slices = {}
print("Searching for case with kidney+tumor slices (best per orientation)...")
for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    data = np.load(os.path.join(PROCESSED_DIR, f))
    mask = np.round(data['mask']).astype(np.uint8)
    # Ensure at least one slice in every orientation has both kidney & tumor
    valid = True
    slices = {}
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = best_slice(mask, orient, kidney_and_tumor=True)
        if sl is None:
            valid = False
            break
        slices[orient] = sl
    if valid:
        case1_found = True
        case1_id = case_id
        case1_slices = slices
        break

# ----------- Find case and slices for kidney-only -----------
case2_found = False
case2_id = None
case2_slices = {}
print("Searching for a different case with kidney-only slices (best per orientation)...")
for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    if case_id == case1_id:
        continue
    data = np.load(os.path.join(PROCESSED_DIR, f))
    mask = np.round(data['mask']).astype(np.uint8)
    valid = True
    slices = {}
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = best_slice(mask, orient, kidney_only=True)
        if sl is None:
            valid = False
            break
        slices[orient] = sl
    if valid:
        case2_found = True
        case2_id = case_id
        case2_slices = slices
        break

# Output 6 images per case (3 overlays, 3 raw) for each case
if case1_found:
    data = np.load(os.path.join(PROCESSED_DIR, f"{case1_id}_vol.npz"))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
    print(f"\n[INFO] Selected {case1_id} for 'kidney_and_tumor' example")
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = case1_slices[orient]
        img2d = img[:, :, sl] if orient=='axial' else img[sl, :, :] if orient=='sagittal' else img[:, sl, :]
        mask2d = mask[:, :, sl] if orient=='axial' else mask[sl, :, :] if orient=='sagittal' else mask[:, sl, :]
        fname_prefix = os.path.join(QC_DIR, f"{case1_id}_kidney_and_tumor_{orient}")
        save_matched_pair(img2d, mask2d, fname_prefix, include_tumor=True)

if case2_found:
    data = np.load(os.path.join(PROCESSED_DIR, f"{case2_id}_vol.npz"))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
    print(f"\n[INFO] Selected {case2_id} for 'kidney_only' example")
    for orient in ['axial', 'sagittal', 'coronal']:
        sl = case2_slices[orient]
        img2d = img[:, :, sl] if orient=='axial' else img[sl, :, :] if orient=='sagittal' else img[:, sl, :]
        mask2d = mask[:, :, sl] if orient=='axial' else mask[sl, :, :] if orient=='sagittal' else mask[:, sl, :]
        fname_prefix = os.path.join(QC_DIR, f"{case2_id}_kidney_only_{orient}")
        save_matched_pair(img2d, mask2d, fname_prefix, include_tumor=False)

print("Done.")
