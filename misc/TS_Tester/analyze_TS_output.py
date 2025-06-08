import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

def dice_score(a, b):
    a = a > 0
    b = b > 0
    intersection = np.logical_and(a, b).sum()
    union = a.sum() + b.sum()
    if union == 0:
        return np.nan
    return 2.0 * intersection / union

def hausdorff(a, b):
    a_pts = np.argwhere(a)
    b_pts = np.argwhere(b)
    if len(a_pts) == 0 or len(b_pts) == 0:
        return np.nan
    return max(
        directed_hausdorff(a_pts, b_pts)[0],
        directed_hausdorff(b_pts, a_pts)[0]
    )

def analyze_masks(kits_dir, ts_dir, output_csv=None):
    kits_files = {os.path.splitext(os.path.basename(f))[0]: f for f in os.listdir(kits_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}
    ts_files = {os.path.splitext(os.path.basename(f))[0]: f for f in os.listdir(ts_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}
    cases = sorted(set(kits_files) & set(ts_files))

    results = []
    for cid in cases:
        gt = nib.load(os.path.join(kits_dir, kits_files[cid])).get_fdata()
        pred = nib.load(os.path.join(ts_dir, ts_files[cid])).get_fdata()
        dice = dice_score(gt, pred)
        hd = hausdorff(gt > 0, pred > 0)
        results.append({'case_id': cid, 'dice': dice, 'hausdorff': hd})

    df = pd.DataFrame(results)
    print(df.describe())
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TotalSegmentator kidney output vs KiTS23 ground truth.")
    parser.add_argument("--kits_mask_dir", type=str, required=True, help="Directory with KiTS23 kidney masks (.nii or .nii.gz)")
    parser.add_argument("--ts_mask_dir", type=str, required=True, help="Directory with TotalSegmentator kidney masks (.nii or .nii.gz)")
    parser.add_argument("--output_csv", type=str, help="CSV file to save per-case results (optional)")
    args = parser.parse_args()
    analyze_masks(args.kits_mask_dir, args.ts_mask_dir, args.output_csv)
