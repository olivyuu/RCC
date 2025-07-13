import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
CSV_PATH = "dev/runs/report_analysis/2025_07_09_15_26_46_evaluation.csv"
OUT_DIR = "dev/data/qc/discrepancy_cases/"
os.makedirs(OUT_DIR, exist_ok=True)

def save_matched_pair(img2d, mask2d, fname_prefix, include_tumor):
    # Raw image
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img2d, cmap='gray')
    ax.axis('off')
    plt.savefig(f"{fname_prefix}_raw.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Overlay
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img2d, cmap='gray')
    if np.any(mask2d == 1):
        ax.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
    if include_tumor and np.any(mask2d == 2):
        ax.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
    ax.axis('off')
    plt.savefig(f"{fname_prefix}_overlay.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

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

df = pd.read_csv(CSV_PATH)
df = df[(df['exam_ground_truth'] == df['exam_prediction']) & (df['exam_ground_truth'] == 1)]

for _, row in df.iterrows():
    case_id = row['exam']
    gt = row['exam_ground_truth']
    data_path = os.path.join(PROCESSED_DIR, f"{case_id}_vol.npz")
    if not os.path.exists(data_path):
        continue
    data = np.load(data_path)
    img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
    for orient in ['axial', 'sagittal', 'coronal']:
        # Always select kidney+tumor images since gt==1 for all cases now
        sl = best_slice(mask, orient, kidney_and_tumor=True)
        include_tumor = True
        if sl is not None:
            img2d = img[:, :, sl] if orient=='axial' else img[sl, :, :] if orient=='sagittal' else img[:, sl, :]
            mask2d = mask[:, :, sl] if orient=='axial' else mask[sl, :, :] if orient=='sagittal' else mask[:, sl, :]
            fname_prefix = os.path.join(OUT_DIR, f"{case_id}_{orient}")
            save_matched_pair(img2d, mask2d, fname_prefix, include_tumor)
    # Save report
    with open(os.path.join(OUT_DIR, f"{case_id}_report.txt"), 'w') as f:
        f.write(row['synthetic_report'])

print("Done.")
