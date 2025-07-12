import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
QC_DIR = "dev/data/qc/three_plane_examples/"
os.makedirs(QC_DIR, exist_ok=True)


def save_three_planes(image, mask, case_id, base_name, colors, mask_label=None):
    planes = {
        'axial': (image.shape[2] // 2, lambda img, sl: img[:, :, sl]),
        'sagittal': (image.shape[0] // 2, lambda img, sl: img[sl, :, :]),
        'coronal': (image.shape[1] // 2, lambda img, sl: img[:, sl, :])
    }
    for plane, (sl, slicer) in planes.items():
        img2d = slicer(image, sl)
        mask2d = slicer(mask, sl)

        # Save raw
        plt.imsave(os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_raw.png"), img2d, cmap='gray')

        # Save kidney only
        kidney_mask = (mask2d == 1).astype(float)
        plt.figure()
        plt.imshow(img2d, cmap='gray')
        plt.imshow(kidney_mask, cmap='Blues', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_kidney.png"))
        plt.close()

        if mask_label:
            overlay_mask = (mask2d == mask_label).astype(float)
            plt.figure()
            plt.imshow(img2d, cmap='gray')
            plt.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
            plt.imshow(overlay_mask, cmap=colors, alpha=0.4)
            plt.axis('off')
            plt.savefig(os.path.join(QC_DIR, f"{case_id}_{base_name}_{plane}_overlay.png"))
            plt.close()


found_tumor = False
found_kidney_only = False

for f in sorted(os.listdir(PROCESSED_DIR)):
    if not f.endswith("_vol.npz"):
        continue

    case_id = f.split("_vol.npz")[0]
    path = os.path.join(PROCESSED_DIR, f)
    data = np.load(path)
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)

    if not found_tumor and np.any(mask == 2):
        save_three_planes(img, mask, case_id, "has_tumor", colors='Reds', mask_label=2)
        found_tumor = True

    elif not found_kidney_only and (1 in np.unique(mask)) and not np.any(mask == 2):
        save_three_planes(img, mask, case_id, "kidney_only", colors=None)
        found_kidney_only = True

    if found_tumor and found_kidney_only:
        break
