import os
import argparse
from glob import glob

def main(kits_mask_dir, ts_mask_dir):
# Assume mask filenames are either case IDs or contain them
kits_files = sorted(glob(os.path.join(kits_mask_dir, ".nii")))
ts_files = sorted(glob(os.path.join(ts_mask_dir, ".nii")))

python
Copy
Edit
kits_ids = {os.path.splitext(os.path.basename(f))[0] for f in kits_files}
ts_ids = {os.path.splitext(os.path.basename(f))[0] for f in ts_files}

common = kits_ids & ts_ids
missing_in_ts = kits_ids - ts_ids
missing_in_kits = ts_ids - kits_ids

print(f"Total KiTS23 kidney masks: {len(kits_files)}")
print(f"Total TotalSegmentator kidney masks: {len(ts_files)}")
print(f"Cases with both: {len(common)}")
print(f"Missing in TotalSegmentator: {sorted(missing_in_ts)}")
print(f"Missing in KiTS23: {sorted(missing_in_kits)}")
if name == "main":
parser = argparse.ArgumentParser(
description="Check correspondence between KiTS23 and TotalSegmentator kidney masks."
)
parser.add_argument(
"--kits_mask_dir", type=str, required=True,
help="Directory with KiTS23 kidney masks (.nii or .nii.gz)"
)
parser.add_argument(
"--ts_mask_dir", type=str, required=True,
help="Directory with TotalSegmentator kidney masks (.nii or .nii.gz)"
)
args = parser.parse_args()
main(args.kits_mask_dir, args.ts_mask_dir)