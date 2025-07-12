import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
CSV_PATH = "dev/runs/report_analysis/2025_07_09_15_26_46_evaluation.csv"
OUT_DIR = "dev/data/qc/discrepancy_cases/"
os.makedirs(OUT_DIR, exist_ok=True)

def save_pair(img2d, mask2d, fname_prefix, gt_has_tumor):
    # Raw
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
    if gt_has_tumor and np.any(mask2d == 2):
        ax.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
    ax.axis('off')
    plt.savefig(f"{fname_prefix}_overlay.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def find_discrepant_slice(mask, orientation, want_tumor_and_kidney, want_kidney_only):
    axis = {'axial':2, 'sagittal':0, 'coronal':1}[orientation]
    n_slices = mask.shape[axis]
    for sl in range(n_slices):
        mask2d = mask[:, :, sl] if orientation=='axial' else mask[sl, :, :] if orientation=='sagittal' else mask[:, sl, :]
        if want_tumor_and_kidney:
            if np.any(mask2d == 1) and np.any(mask2d == 2):
                return sl
        if want_kidney_only:
            if np.any(mask2d == 1) and not np.any(mask2d == 2) and not np.any(mask2d == 3):
                return sl
    return None

df = pd.read_csv(CSV_PATH)
df = df[(df['exam_ground_truth'] == df['exam_prediction']) & (df['overall_discrepancy'] == 1)]

needed_orients = ['axial']*3 + ['sagittal']*3 + ['coronal']*3
example_idx = 0
used_cases = set()

for orient in needed_orients:
    found = False
    for idx, row in df.iterrows():
        case_id = row['exam']
        if (case_id, orient) in used_cases:
            continue
        gt = row['exam_ground_truth']
        data_path = os.path.join(PROCESSED_DIR, f"{case_id}_vol.npz")
        if not os.path.exists(data_path):
            continue
        data = np.load(data_path)
        img, mask = data['image'], np.round(data['mask']).astype(np.uint8)
        if gt == 1:
            sl = find_discrepant_slice(mask, orient, want_tumor_and_kidney=True, want_kidney_only=False)
        else:
            sl = find_discrepant_slice(mask, orient, want_tumor_and_kidney=False, want_kidney_only=True)
        if sl is not None:
            img2d = img[:, :, sl] if orient=='axial' else img[sl, :, :] if orient=='sagittal' else img[:, sl, :]
            mask2d = mask[:, :, sl] if orient=='axial' else mask[sl, :, :] if orient=='sagittal' else mask[:, sl, :]
            fname_prefix = os.path.join(OUT_DIR, f"{case_id}_{orient}")
            save_pair(img2d, mask2d, fname_prefix, gt_has_tumor=(gt==1))
            # Save report
            with open(os.path.join(OUT_DIR, f"{case_id}_{orient}_report.txt"), 'w') as f:
                f.write(row['synthetic_report'])
            used_cases.add((case_id, orient))
            print(f"Saved images and report for case {case_id} ({orient})")
            example_idx += 1
            found = True
            break
    if not found:
        print(f"No valid example found for orientation: {orient}")

print("Done.")
