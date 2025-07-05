# Kidney Lesion Detection on KiTS23

*Patch-based AI detection of suspicious kidney lesions on CT*

---

## Project Structure & File Organization

* **`kits23/dataset/`** → *Place your raw KiTS23 data here* (`case_xxxxx/imaging.nii.gz`, `segmentation.nii.gz`)
* **`dev/data/processed/kits23/`** → *Preprocessed patch/volume files* (auto-created)
* **`dev/runs/`** → *Model outputs and logs* (auto-created for each training run)
* **`dev/scripts/` or `dev/detection/`** → *Core scripts and modules* (dataset, model, trainer, etc.)

---

## 1. Setup and Dataset Preparation

### A. Download/Place the KiTS23 Dataset

1. **Download** from [kits-challenge.org](https://kits-challenge.org/)
2. **Extract** so that you have a folder like:

   ```
   kits23/dataset/case_00001/imaging.nii.gz
   kits23/dataset/case_00001/segmentation.nii.gz
   ...
   ```

### B. Preprocessing — Create Patches from Volumes

**Run:**

```bash
python dev/detection/preprocessing.py --raw_dir kits23/dataset/
```

*Optionally*, use `--max_cases` to process a subset for testing:

```bash
python dev/detection/preprocessing.py --raw_dir kits23/dataset/ --max_cases 10
```

**Output:**

* For each patient:

  * `dev/data/processed/kits23/case_xxxxx_patches.npz` (patches & labels)
  * `dev/data/processed/kits23/case_xxxxx_vol.npz` (whole resized volume, for QC)

---

## 2. Training a Detection Model

### A. Configure Experiment

Edit or copy `dev/config/detection.yaml`.
**Key settings:**

* `arch`: Choose `"densenet121"` (recommended), or `"densenet169"`
* `epochs`, `batch_size`, `dropout`, learning rate schedule, etc.

### B. Launch Training

**Standard training:**

```bash
python dev/scripts/train_detection.py --config dev/config/detection.yaml
```

* Outputs will be saved to a new folder under `dev/runs/` (e.g. `dev/runs/run_20240704_133217/`)
* Training logs, config, and best model checkpoint (`best_model.pt`) are stored there.

**Testing pipeline (5 epochs, small batch):**

```bash
python dev/scripts/train_detection.py --config dev/config/detection.yaml --testing
```

---

## 3. Evaluation and QC

### A. Evaluate a Trained Model

**After training, run:**

```bash
python dev/scripts/eval_detection.py \
  --model_path dev/runs/<RUN_FOLDER>/best_model.pt \
  --run_dir dev/runs/<RUN_FOLDER> \
  --processed_dir dev/data/processed/kits23/ \
  --split val \
  --num_examples 10
```

* Replace `<RUN_FOLDER>` with your latest run folder name.

**What does this do?**

* Evaluates model on val set, computes metrics
* Outputs to `<RUN_FOLDER>/qc/`:

  * `metrics.txt`
  * `roc_curve.png`
  * `confusion_matrix.png`
  * Example patch visualizations (confident positive/negative, uncertain)

### B. QC Scripts

#### Patch-level Dataset QC

```bash
python dev/scripts/validate_dataset.py --processed_dir dev/data/processed/kits23/ --split train
python dev/scripts/validate_dataset.py --processed_dir dev/data/processed/kits23/ --split val
```

* Prints class balance, shows meta for sanity checking, and saves example patches to `dev/data/qc/patches/`

#### Preprocessing QC (Overlay Masks)

```bash
python dev/scripts/validate_preprocessing.py --processed_dir dev/data/processed/kits23/ --qc_dir dev/data/qc/kits23/
```

* Saves overlay images of original volumes and masks (kidney/tumor/cyst) for quick human inspection.

---

## 4. File/Folder Organization Details

* **Raw Data:**

  * `kits23/dataset/case_*/imaging.nii.gz`, `segmentation.nii.gz`
* **Preprocessed Data:**

  * `dev/data/processed/kits23/case_00001_patches.npz` (all patches for a case)
  * `dev/data/processed/kits23/case_00001_vol.npz` (whole volume)
* **Runs:**

  * `dev/runs/run_<timestamp>/` — each run gets its own folder, storing:

    * `train.log`
    * `config.yaml`
    * `best_model.pt`
    * `qc/` (metrics, curves, patch images)

---

## 5. Notes & Troubleshooting

* **Splitting:** The split is patient-wise, not patch-wise—so no patient is in both train and val.
* **Augmentation:** Controlled via `detection.yaml` or can be modified in `dataset.py`
* **Config-driven:** Nearly all experiment settings are configurable via YAML.
* **QC first:** Always validate both preprocessing and dataset splits using the QC scripts before training.

---

## 6. Example: Full Pipeline

1. **Preprocess data:**

   ```bash
   python dev/detection/preprocessing.py --raw_dir kits23/dataset/
   ```
2. **Train:**

   ```bash
   python dev/scripts/train_detection.py --config dev/config/detection.yaml
   ```
3. **Eval:**

   ```bash
   python dev/scripts/eval_detection.py --model_path dev/runs/<RUN_FOLDER>/best_model.pt --run_dir dev/runs/<RUN_FOLDER> --processed_dir dev/data/processed/kits23/ --split val --num_examples 10
   ```
4. **QC (optional, anytime):**

   ```bash
   python dev/scripts/validate_dataset.py --processed_dir dev/data/processed/kits23/ --split val
   python dev/scripts/validate_preprocessing.py --processed_dir dev/data/processed/kits23/ --qc_dir dev/data/qc/kits23/
   ```

---

## 7. Contact

For questions, open an issue or contact the repo maintainer.

---
