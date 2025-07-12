import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
QC_DIR = "dev/data/qc/three_plane_examples/"
os.makedirs(QC_DIR, exist_ok=True)

def save_six_views(image, mask, case_id, base_name, include_tumor_mask=True):
    """
    For each plane, save a raw (no mask) and an overlay (kidney+/-tumor) image.
    """
    planes = {
        'axial': (image.shape[2] // 2, lambda img, sl: img[:, :, sl], 2),
        'sagittal': (image.shape[0] // 2, lambda img, sl: img[sl, :, :], 0),
        'coronal': (image.shape[1] // 2, lambda img, sl: img[:, sl, :], 1)
    }
    for plane, (sl, slicer, axis) in planes.items():
        img2d = slicer(image, sl)
        mask2d = slicer(mask, sl)

        # 1. Raw image
        plt.imsave(os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_raw.png"), img2d, cmap='gray')

        # 2. Overlay
        plt.figure()
        plt.imshow(img2d, cmap='gray')
        if np.any(mask2d == 1):
            plt.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
        if include_tumor_mask and np.any(mask2d == 2):
            plt.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_overlay.png"))
        plt.close()

# --------- Find two cases ---------
found_both = False
found_kidney = False

for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    data = np.load(os.path.join(PROCESSED_DIR, f))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)

    # Case 1: kidney and tumor (any slice with both present, not necessarily overlapping pixels)
    has_kidney = np.any(mask == 1)
    has_tumor = np.any(mask == 2)
    if not found_both and has_kidney and has_tumor:
        save_six_views(img, mask, case_id, "kidney_and_tumor", include_tumor_mask=True)
        found_both = True
        continue

    # Case 2: only kidney (no tumor present at all)
    if not found_kidney and has_kidney and not has_tumor:
        save_six_views(img, mask, case_id, "kidney_only", include_tumor_mask=False)
        found_kidney = True
        continue

    if found_both and found_kidney:
        break
