import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
QC_DIR = "dev/data/qc/three_plane_examples/"
os.makedirs(QC_DIR, exist_ok=True)

def save_six_views(image, mask, case_id, base_name, include_tumor_mask=True):
    """
    For each plane, save a raw (no mask) and an overlay (kidney+/-tumor) image, 
    ensuring images are dimension-matched and have consistent axis settings.
    """
    planes = {
        'axial': (image.shape[2] // 2, lambda img, sl: img[:, :, sl], 2),
        'sagittal': (image.shape[0] // 2, lambda img, sl: img[sl, :, :], 0),
        'coronal': (image.shape[1] // 2, lambda img, sl: img[:, sl, :], 1)
    }
    for plane, (sl, slicer, axis) in planes.items():
        img2d = slicer(image, sl)
        mask2d = slicer(mask, sl)
        
        # Create a common figure size for all images
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img2d, cmap='gray')
        ax.axis('off')
        raw_path = os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_raw.png")
        plt.savefig(raw_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved raw image: {raw_path}")

        # Overlay
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img2d, cmap='gray')
        mask_drawn = False
        if np.any(mask2d == 1):
            ax.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
            mask_drawn = True
        if include_tumor_mask and np.any(mask2d == 2):
            ax.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
            mask_drawn = True
        ax.axis('off')
        overlay_path = os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_overlay.png")
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        if mask_drawn:
            print(f"Saved overlay image: {overlay_path}")
        else:
            print(f"Saved overlay image (no mask present in this slice): {overlay_path}")

# --------- Find two cases ---------
found_both = False
found_kidney = False

print("Starting search for appropriate cases...")

for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue
    case_id = f.split("_vol.npz")[0]
    data = np.load(os.path.join(PROCESSED_DIR, f))
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)

    has_kidney = np.any(mask == 1)
    has_tumor = np.any(mask == 2)
    if not found_both and has_kidney and has_tumor:
        print(f"\n[INFO] Selected {case_id} for 'kidney_and_tumor' example")
        save_six_views(img, mask, case_id, "kidney_and_tumor", include_tumor_mask=True)
        found_both = True

    if not found_kidney and has_kidney and not has_tumor:
        print(f"\n[INFO] Selected {case_id} for 'kidney_only' example")
        save_six_views(img, mask, case_id, "kidney_only", include_tumor_mask=False)
        found_kidney = True

    if found_both and found_kidney:
        print("\nDone! Both cases processed successfully.")
        break
