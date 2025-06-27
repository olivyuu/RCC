import os
import numpy as np
import matplotlib.pyplot as plt

def plot_mask_overlay(image, mask, mask_name, outpath):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    if mask is not None and np.any(mask):
        plt.imshow(mask, alpha=0.4, cmap='Reds')
        plt.title(f"{mask_name} Overlay")
    else:
        plt.title("Original Image" if mask is None else f"{mask_name} (Not present)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved: {outpath}")
    plt.close()

def validate_preprocessing(processed_dir, qc_dir, max_cases=5, ensure_cyst=True):
    os.makedirs(qc_dir, exist_ok=True)
    vol_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('_vol.npz')])

    found_cyst = False
    checked = 0
    for vol_file in vol_files:
        if checked >= max_cases and (not ensure_cyst or found_cyst):
            break

        try:
            data = np.load(os.path.join(processed_dir, vol_file))
            image = data['image']
            mask = data['mask']
        except Exception as e:
            print(f"[WARN] Could not load {vol_file}: {e}. File may be corrupted, incomplete, or not a valid .npz. Consider deleting and reprocessing.")
            continue

        # Fix mask: round and convert to uint8 (should have already been done in preprocessor, but just in case)
        mask = np.round(mask).astype(np.uint8)

        case_id = vol_file.split('_')[0]

        # Print basic info for QC
        print(f"\n=== QC for {case_id} ===")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}, min/max: {image.min():.3f}/{image.max():.3f}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, mask labels: {np.unique(mask)}")

        # 1. Save a mid-slice with just original image (for reference)
        mid = image.shape[2] // 2
        out_orig = os.path.join(qc_dir, f"{case_id}_orig.png")
        plot_mask_overlay(image[:,:,mid], None, "Original", out_orig)

        # 2. Find a slice with kidney and tumor present, save that
        kidney_slices = np.unique(np.argwhere(mask == 1)[:, 2])
        tumor_slices = np.unique(np.argwhere(mask == 2)[:, 2])
        common_slices = np.intersect1d(kidney_slices, tumor_slices)
        slice_to_plot = None
        if len(common_slices) > 0:
            slice_to_plot = int(common_slices[0])
            print(f"  [INFO] Slice {slice_to_plot}: kidney and tumor present.")
        elif len(kidney_slices) > 0:
            slice_to_plot = int(kidney_slices[0])
            print(f"  [INFO] Slice {slice_to_plot}: kidney present, tumor not present.")
        else:
            slice_to_plot = mid
            print("  [INFO] Defaulting to mid-slice.")

        # Save kidney and tumor overlay for selected slice
        out_kidney = os.path.join(qc_dir, f"{case_id}_kidney.png")
        plot_mask_overlay(
            image[:,:,slice_to_plot],
            (mask[:,:,slice_to_plot] == 1).astype(float),
            "Kidney", out_kidney
        )
        out_tumor = os.path.join(qc_dir, f"{case_id}_tumor.png")
        plot_mask_overlay(
            image[:,:,slice_to_plot],
            (mask[:,:,slice_to_plot] == 2).astype(float),
            "Tumor", out_tumor
        )

        # 3. If cyst present, find a slice with cyst and save it
        cyst_slices = np.unique(np.argwhere(mask == 3)[:, 2])
        if len(cyst_slices) > 0:
            cyst_slice = int(cyst_slices[0])
            out_cyst = os.path.join(qc_dir, f"{case_id}_cyst.png")
            plot_mask_overlay(
                image[:,:,cyst_slice],
                (mask[:,:,cyst_slice] == 3).astype(float),
                "Cyst", out_cyst
            )
            print(f"  [INFO] Cyst found in slice {cyst_slice}.")
            found_cyst = True
        else:
            print(f"  No cyst present for {case_id}.")

        checked += 1

    if ensure_cyst and not found_cyst:
        print(f"Warning: No cyst mask found in the first {max_cases} cases.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='dev/data/processed/kits23/')
    parser.add_argument('--qc_dir', type=str, default='dev/data/qc/kits23/')
    parser.add_argument('--max_cases', type=int, default=10)
    parser.add_argument('--ensure_cyst', action='store_true', help='Try to include at least one case with a cyst mask in QC')
    args = parser.parse_args()
    validate_preprocessing(
        args.processed_dir, 
        args.qc_dir, 
        max_cases=args.max_cases, 
        ensure_cyst=args.ensure_cyst
    )
