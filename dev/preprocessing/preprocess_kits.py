import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def resample_image(img, out_spacing, is_label=False):
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_size = [
        int(np.round(i * s / o))
        for i, s, o in zip(in_size, in_spacing, out_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resampler.Execute(img)

def intensity_preprocess(np_img, min_hu=-200, max_hu=300):
    np_img = np.clip(np_img, min_hu, max_hu)
    np_img = (np_img - min_hu) / (max_hu - min_hu)
    return np_img.astype(np.float32)

def crop_to_mask(np_img, np_mask, margin=10):
    # Crop the image and mask to the bounding box of the mask (+margin)
    coords = np.array(np.where(np_mask > 0))
    if coords.size == 0:
        # Empty mask; return the original arrays
        return np_img, np_mask
    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1)
    minz = max(minz - margin, 0)
    miny = max(miny - margin, 0)
    minx = max(minx - margin, 0)
    maxz = min(maxz + margin, np_mask.shape[0] - 1)
    maxy = min(maxy + margin, np_mask.shape[1] - 1)
    maxx = min(maxx + margin, np_mask.shape[2] - 1)
    cropped_img = np_img[minz:maxz+1, miny:maxy+1, minx:maxx+1]
    cropped_mask = np_mask[minz:maxz+1, miny:maxy+1, minx:maxx+1]
    return cropped_img, cropped_mask

def center_pad_or_crop(arr, target_shape):
    """Center pad or crop a 3D array to the target shape."""
    res = np.zeros(target_shape, dtype=arr.dtype)
    in_shape = np.array(arr.shape)
    tgt_shape = np.array(target_shape)
    # Calculate start and end indices for both
    offset_in = np.maximum((in_shape - tgt_shape) // 2, 0)
    offset_out = np.maximum((tgt_shape - in_shape) // 2, 0)
    crop_shape = np.minimum(in_shape, tgt_shape)
    in_slices = [slice(offset_in[i], offset_in[i] + crop_shape[i]) for i in range(3)]
    out_slices = [slice(offset_out[i], offset_out[i] + crop_shape[i]) for i in range(3)]
    res[out_slices[0], out_slices[1], out_slices[2]] = arr[in_slices[0], in_slices[1], in_slices[2]]
    return res

def process_case(case_dir, out_img_dir, out_mask_dir, spacing, separate_masks=False, target_shape=(128,128,128)):
    case_id = os.path.basename(case_dir)
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")
    if not (os.path.isfile(img_path) and os.path.isfile(seg_path)):
        print(f"Warning: Missing files for {case_id}")
        return

    # Read and resample
    img = sitk.ReadImage(img_path)
    img = resample_image(img, spacing, is_label=False)
    img_np = sitk.GetArrayFromImage(img)

    seg = sitk.ReadImage(seg_path)
    seg = resample_image(seg, spacing, is_label=True)
    seg_np = sitk.GetArrayFromImage(seg)

    # Preprocess intensities and crop to mask
    img_np = intensity_preprocess(img_np)
    img_np, seg_np = crop_to_mask(img_np, seg_np, margin=10)

    # Pad/crop to target shape
    img_np = center_pad_or_crop(img_np, target_shape)
    seg_np = center_pad_or_crop(seg_np, target_shape)

    # Save output
    out_img_path = os.path.join(out_img_dir, f"{case_id}.nii.gz")
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img_sitk, out_img_path)

    if separate_masks:
        # Save kidney, tumor, cyst masks separately
        for label, name in zip([1, 2, 3], ['kidney', 'tumor', 'cyst']):
            mask = (seg_np == label).astype(np.uint8)
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.SetSpacing(tuple(float(s) for s in spacing))
            out_mask_path = os.path.join(out_mask_dir, name, f"{case_id}.nii.gz")
            os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
            sitk.WriteImage(mask_sitk, out_mask_path)
    else:
        out_mask_path = os.path.join(out_mask_dir, f"{case_id}.nii.gz")
        mask_sitk = sitk.GetImageFromArray(seg_np.astype(np.uint8))
        mask_sitk.SetSpacing(tuple(float(s) for s in spacing))
        sitk.WriteImage(mask_sitk, out_mask_path)

def main(kits_root, out_dir, spacing, separate_masks, target_shape):
    out_img_dir = os.path.join(out_dir, "images")
    out_mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    case_dirs = [os.path.join(kits_root, d) for d in os.listdir(kits_root) if d.startswith("case_")]
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        process_case(case_dir, out_img_dir, out_mask_dir, spacing, separate_masks, target_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KiTS23 dataset for deep learning.")
    parser.add_argument("--kits_root", required=True, help="Path to KiTS23 dataset root (case_xxxxx subfolders)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5,1.5,3.0], help="Resample spacing (x y z, mm)")
    parser.add_argument("--separate_masks", action="store_true", help="Save separate kidney/tumor/cyst masks")
    parser.add_argument("--shape", type=int, nargs=3, default=[128,128,128], help="Output shape (z y x)")
    args = parser.parse_args()
    main(args.kits_root, args.out_dir, tuple(args.spacing), args.separate_masks, tuple(args.shape))
