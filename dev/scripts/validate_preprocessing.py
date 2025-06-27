import os
import numpy as np
import matplotlib.pyplot as plt

def plot_multiclass_overlay(image, mask, mask_names, outpath, colors, priorities):
    """Overlay multiple binary masks in color on a grayscale image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    overlayed = False
    for idx in priorities:
        this_mask = mask[idx]
        if this_mask is not None and np.any(this_mask):
            plt.imshow(this_mask, alpha=0.4, cmap=colors[idx])
            overlayed = True
    if overlayed:
        legend = [plt.Rectangle((0,0),1,1, color=plt.get_cmap(colors[idx])(0.5)) for idx in priorities if mask[idx] is not None and np.any(mask[idx])]
        legend_labels = [mask_names[idx] for idx in priorities if mask[idx] is not None and np.any(mask[idx])]
        plt.legend(legend, legend_labels, loc='lower right', fontsize=8)
        plt.title("Overlay: " + " + ".join(legend_labels))
    else:
        plt.title("Original Image (no mask found)")
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

        mask = np.round(mask).astype(np.uint8)
        # Use the first two tokens for case_id: e.g. 'case_00123'
        case_id = '_'.join(vol_file.split('_')[:2])

        print(f"\n=== QC for {case_id} ===")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}, min/max: {image.min():.3f}/{image.max():.3f}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, mask labels: {np.unique(mask)}")

        # Save original mid-slice for reference
        mid = image.shape[2] // 2
        out_orig = os.path.join(qc_dir, f"{case_id}_orig_{mid}.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(image[:,:,mid], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_orig)
        print(f"Saved: {out_orig}")
        plt.close()

        cyst_slices = np.unique(np.argwhere(mask == 3)[:, 2])
        if len(cyst_slices) > 0:
            # Show a cyst + kidney overlay
            slice_to_plot = int(cyst_slices[0])
            mask_kidney = (mask[:,:,slice_to_plot] == 1).astype(float)
            mask_cyst = (mask[:,:,slice_to_plot] == 3).astype(float)
            # Cyst overlays kidney
            mask_list = [mask_kidney, mask_cyst]
            mask_names = ["Kidney", "Cyst"]
            colors = ["Blues", "Oranges"] # Oranges = cyst, on top of blue kidney
            priorities = [0, 1]
            outpath = os.path.join(qc_dir, f"{case_id}_kidney_cyst_overlay_{slice_to_plot}.png")
            plot_multiclass_overlay(
                image[:,:,slice_to_plot], mask_list, mask_names, outpath, colors, priorities
            )
            print(f"  [INFO] Slice {slice_to_plot}: cyst and kidney present (cyst overlays kidney).")
            found_cyst = True
        else:
            # No cyst, find kidney+tumor
            kidney_slices = np.unique(np.argwhere(mask == 1)[:, 2])
            tumor_slices = np.unique(np.argwhere(mask == 2)[:, 2])
            common_slices = np.intersect1d(kidney_slices, tumor_slices)
            if len(common_slices) > 0:
                slice_to_plot = int(common_slices[0])
                print(f"  [INFO] Slice {slice_to_plot}: kidney and tumor present (tumor overlays kidney).")
            elif len(kidney_slices) > 0:
                slice_to_plot = int(kidney_slices[0])
                print(f"  [INFO] Slice {slice_to_plot}: only kidney present.")
            else:
                slice_to_plot = mid
                print("  [INFO] Defaulting to mid-slice.")

            mask_kidney = (mask[:,:,slice_to_plot] == 1).astype(float)
            mask_tumor = (mask[:,:,slice_to_plot] == 2).astype(float)
            mask_list = [mask_kidney, mask_tumor]
            mask_names = ["Kidney", "Tumor"]
            colors = ["Blues", "Reds"] # Tumor = red, overlays kidney
            priorities = [0, 1]
            outpath = os.path.join(qc_dir, f"{case_id}_kidney_tumor_overlay_{slice_to_plot}.png")
            plot_multiclass_overlay(
                image[:,:,slice_to_plot], mask_list, mask_names, outpath, colors, priorities
            )

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
