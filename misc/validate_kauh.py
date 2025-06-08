import os
import SimpleITK as sitk
import numpy as np

def main(root):
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    n_cases = 0
    n_with_tumor = 0
    bad_cases = []
    label_set = set()
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        try:
            img = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)
            mask_np = sitk.GetArrayFromImage(mask)
            labels = np.unique(mask_np)
            label_set.update(labels.tolist())
            tumor_present = (mask_np > 1).any()  # update label logic if needed
            n_cases += 1
            if tumor_present:
                n_with_tumor += 1
        except Exception as e:
            bad_cases.append((fname, str(e)))
    print("====== KAUH DATASET VALIDATION ======")
    print(f"Total cases: {n_cases}")
    print(f"Cases with tumor: {n_with_tumor}")
    print(f"Label set across dataset: {label_set}")
    if bad_cases:
        print("\nCorrupted or unreadable cases:")
        for c, err in bad_cases:
            print(f"  {c}: {err}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory of KAUH dataset")
    args = parser.parse_args()
    main(args.root)
