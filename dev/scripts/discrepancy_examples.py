import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "dev/data/processed/kits23/"
CSV_PATH = "dev/runs/report_analysis/2025_07_09_15_26_46_evaluation.csv"
OUT_DIR = "dev/data/qc/discrepancy_cases/"
os.makedirs(OUT_DIR, exist_ok=True)

def save_case_image_and_report(case_id, model_says_tumor, report_text):
    path = os.path.join(PROCESSED_DIR, f"{case_id}_vol.npz")
    if not os.path.exists(path):
        print(f"[WARN] {case_id} not found in processed_dir.")
        return

    data = np.load(path)
    img = data['image']
    mask = np.round(data['mask']).astype(np.uint8)

    # Choose a slice: if model_says_tumor, show a tumor slice; otherwise, show a mid kidney slice
    if model_says_tumor and np.any(mask == 2):
        # Use the first tumor slice
        tumor_slices = np.unique(np.argwhere(mask == 2)[:, 2])
        sl = int(tumor_slices[0]) if len(tumor_slices) > 0 else img.shape[2] // 2
    else:
        # Show a mid kidney slice (with kidney, no tumor)
        kidney_slices = np.unique(np.argwhere(mask == 1)[:, 2])
        sl = int(kidney_slices[len(kidney_slices)//2]) if len(kidney_slices) > 0 else img.shape[2] // 2

    kidney = (mask[:,:,sl] == 1).astype(float)
    tumor = (mask[:,:,sl] == 2).astype(float)

    plt.figure()
    plt.imshow(img[:,:,sl], cmap='gray')
    plt.imshow(kidney, cmap='Blues', alpha=0.4)
    if tumor.any():
        plt.imshow(tumor, cmap='Reds', alpha=0.4)
    plt.title(f"{case_id} Slice {sl}")
    plt.axis('off')
    out_img = os.path.join(OUT_DIR, f"{case_id}_slice{sl}.png")
    plt.savefig(out_img)
    plt.close()
    print(f"Saved: {out_img}")

    out_txt = os.path.join(OUT_DIR, f"{case_id}_report.txt")
    with open(out_txt, 'w') as f:
        f.write(report_text)
    print(f"Saved: {out_txt}")

def main():
    df = pd.read_csv(CSV_PATH)

    # Type A: Report says NO tumor, model says YES tumor
    a_row = df[(df['report_prediction'] == 0) & (df['exam_prediction'] == 1)]
    if not a_row.empty:
        row = a_row.iloc[0]
        case_id = row['exam']
        report_text = row['synthetic_report']
        save_case_image_and_report(case_id, model_says_tumor=True, report_text=report_text)
    else:
        print("[INFO] No cases where report missed tumor but model found tumor.")

    # Type B: Report says YES tumor, model says NO tumor
    b_row = df[(df['report_prediction'] == 1) & (df['exam_prediction'] == 0)]
    if not b_row.empty:
        row = b_row.iloc[0]
        case_id = row['exam']
        report_text = row['synthetic_report']
        save_case_image_and_report(case_id, model_says_tumor=False, report_text=report_text)
    else:
        print("[INFO] No cases where report called tumor but model said no tumor.")

if __name__ == "__main__":
    main()
