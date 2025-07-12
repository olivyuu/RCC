import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
CSV_PATH = "dev/runs/report_analysis/2025_07_09_15_26_46_evaluation.csv"
OUT_DIR = "dev/data/qc/discrepancy_cases/"
os.makedirs(OUT_DIR, exist_ok=True)

def visualize_case(case_id, slice_hint=None):
    path = os.path.join(PROCESSED_DIR, f"{case_id}_vol.npz")
    if not os.path.exists(path):
        print(f"[WARN] {case_id} not found in processed_dir.")
        return

    data = np.load(path)
    img = data['image']
    mask = np.round(data['mask']).astype(np.uint8)

    if slice_hint is not None and 0 <= slice_hint < img.shape[2]:
        sl = slice_hint
    else:
        tumor_slices = np.where(mask == 2)[2] if mask.ndim == 3 else []
        sl = tumor_slices[0] if len(tumor_slices) else img.shape[2] // 2

    kidney = (mask[:,:,sl] == 1).astype(float)
    tumor = (mask[:,:,sl] == 2).astype(float)

    plt.figure()
    plt.imshow(img[:,:,sl], cmap='gray')
    plt.imshow(kidney, cmap='Blues', alpha=0.4)
    if tumor.any():
        plt.imshow(tumor, cmap='Reds', alpha=0.4)
    plt.title(f"{case_id} Slice {sl}")
    plt.axis('off')
    outpath = os.path.join(OUT_DIR, f"{case_id}_slice{sl}.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")

def main():
    df = pd.read_csv(CSV_PATH)
    case_col = 'case_id' if 'case_id' in df.columns else 'CaseID'
    slice_col = 'slice' if 'slice' in df.columns else None

    for i, row in df.iterrows():
        cid = str(row[case_col])
        sl_hint = int(row[slice_col]) if slice_col and not pd.isna(row[slice_col]) else None
        visualize_case(cid, sl_hint)

if __name__ == "__main__":
    main()