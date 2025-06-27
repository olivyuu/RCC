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

        mid = image.shape[2] // 2
        case_id = vol_file.split('_')[0]

        # Print basic info for QC
        print(f"\n=== QC for {case_id} ===")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}, min/max: {image.min():.3f}/{image.max():.3f}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, mask labels: {np.unique(mask)}")

        # Original image
        plot_mask_overlay(image[:,:,mid], None, "Original", os.path.join(qc_dir, f"{case_id}_orig.png"))
        # Kidney mask
        plot_mask_overlay(image[:,:,mid], (mask[:,:,mid] == 1).astype(float), "Kidney", os.path.join(qc_dir, f"{case_id}_kidney.png"))
        # Tumor mask
        plot_mask_overlay(image[:,:,mid], (mask[:,:,mid] == 2).astype(float), "Tumor", os.path.join(qc_dir, f"{case_id}_tumor.png"))
        # Cyst mask (only if present)
        cyst_mask = (mask[:,:,mid] == 3)
        if np.any(cyst_mask):
            plot_mask_overlay(image[:,:,mid], cyst_mask.astype(float), "Cyst", os.path.join(qc_dir, f"{case_id}_cyst.png"))
            found_cyst = True
        else:
            print(f"No cyst in mid-slice for {case_id}")

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
