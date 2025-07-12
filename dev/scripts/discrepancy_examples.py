import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

PROCESSED_DIR = "dev/data/processed/kits23/"
CSV_PATH = "dev/runs/report_analysis/2025_07_09_15_26_46_evaluation.csv"
OUT_DIR = "dev/data/qc/discrepancy_cases/"
os.makedirs(OUT_DIR, exist_ok=True)

def save_case_images(case_id, report_text, random_plane):
    path = os.path.join(PROCESSED_DIR, f"{case_id}_vol.npz")
    if not os.path.exists(path):
        print(f"[WARN] {case_id} not found in processed_dir.")
        return

    data = np.load(path)
    img = data['image']
    mask = np.round(data['mask']).astype(np.uint8)

    planes = {
        'axial': (img.shape[2] // 2, lambda x, sl: x[:, :, sl]),
        'sagittal': (img.shape[0] // 2, lambda x, sl: x[sl, :, :]),
        'coronal': (img.shape[1] // 2, lambda x, sl: x[:, sl, :])
    }
    sl, slicer = planes[random_plane]
    img2d = slicer(img, sl)
    mask2d = slicer(mask, sl)

    # Raw image
    raw_path = os.path.join(OUT_DIR, f"{case_id}_{random_plane}_raw.png")
    plt.imsave(raw_path, img2d, cmap='gray')

    # Overlay image
    plt.figure()
    plt.imshow(img2d, cmap='gray')
    if np.any(mask2d == 1):
        plt.imshow((mask2d == 1), cmap='Blues', alpha=0.4)
    if np.any(mask2d == 2):
        plt.imshow((mask2d == 2), cmap='Reds', alpha=0.4)
    plt.axis('off')
    overlay_path = os.path.join(OUT_DIR, f"{case_id}_{random_plane}_overlay.png")
    plt.savefig(overlay_path)
    plt.close()
    print(f"Saved: {raw_path}, {overlay_path}")

    # Save report
    out_txt = os.path.join(OUT_DIR, f"{case_id}_report.txt")
    with open(out_txt, 'w') as f:
        f.write(report_text)

def main():
    df = pd.read_csv(CSV_PATH)
    if len(df) < 10:
        print(f"Warning: Only {len(df)} examples in discrepancy file.")
    sample_df = df.sample(n=min(10, len(df)), random_state=42)

    for idx, row in sample_df.iterrows():
        case_id = row['exam']
        report_text = row['synthetic_report']
        random_plane = random.choice(['axial', 'sagittal', 'coronal'])
        save_case_images(case_id, report_text, random_plane)

if __name__ == "__main__":
    main()
