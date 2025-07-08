import os
import argparse
import torch
import numpy as np
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from dev.detection.model import get_model
from dev.detection.dataset import RCCPatchDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--processed_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_examples', type=int, default=10)
    args = parser.parse_args()

    qc_dir = os.path.join(args.run_dir, 'qc')
    os.makedirs(qc_dir, exist_ok=True)

    print(f"Loading {args.split} dataset from {args.processed_dir} ...")
    dataset = RCCPatchDataset(args.processed_dir, split=args.split, augment=False)
    print("Number of patch files found:", len(dataset.patch_files))
    print("Selected files for this split:", len(dataset.selected_files))
    print("Selected file names:", dataset.selected_files)
    print("Total number of patches:", len(dataset))

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(pretrained=False, dropout=None).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    all_preds = []
    all_labels = []
    all_meta = []
    all_probs = []
    all_fileinfo = []
    all_imgs = []
    all_masks = []

    with torch.no_grad():
        for batch_idx, (images, labels, masks, metas) in enumerate(loader):
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_meta.extend(metas)
            # File/case info
        # ---- FIX: Ensure metas is always a list ----
        # Some DataLoader configs may give a dict, some a list. We always want a list of dicts/strs.
            # Ensure metas is a list of meta dicts/strs, one per sample in the batch
            if isinstance(metas, np.ndarray):
                metas = metas.tolist()
            elif isinstance(metas, dict):
                metas = [metas]
            elif isinstance(metas, (list, tuple)):
                metas = list(metas)
            else:
                metas = [metas] * images.shape[0]  # fallback, rare
            print(f"[BATCH {batch_idx}] batch size: {images.shape[0]}, len(metas): {len(metas)}")

            for i in range(len(metas)):
                meta = metas[i]
                patch_file = None
                if isinstance(meta, dict):
                    case = meta.get('case_id', None)
                    slice_num = meta.get('slice', None)
                    patch_type = meta.get('type', None)
                    patch_file = meta.get('patch_file', None)
                elif isinstance(meta, str) and meta.startswith("{") and meta.endswith("}"):
                    import ast
                    try:
                        meta_dict = ast.literal_eval(meta)
                        case = meta_dict.get('case_id', None)
                        slice_num = meta_dict.get('slice', None)
                        patch_type = meta_dict.get('type', None)
                        patch_file = meta_dict.get('patch_file', None)
                    except Exception:
                        case, slice_num, patch_type, patch_file = None, None, None, None
                else:
                    case, slice_num, patch_type, patch_file = None, None, None, None
                all_fileinfo.append((patch_file, slice_num, patch_type, batch_idx, i))



            # For image saving (keep for future reactivation)
            imgs_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(imgs_np.shape[0]):
                img = imgs_np[i,0] if imgs_np[i].ndim == 3 else imgs_np[i]
                all_imgs.append(img)
                mask = masks_np[i,0] if masks_np[i].ndim == 3 else masks_np[i]
                all_masks.append(mask)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Save all predictions as CSV
    csv_path = os.path.join(qc_dir, "predictions.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patch_file", "slice", "patch_type", "batch_idx", "idx_in_batch", "ground_truth", "pred", "confidence"])
        for info, gt, pred, conf in zip(all_fileinfo, all_labels, all_preds, all_probs):
            patch_file, slice_num, patch_type, batch_idx, i = info
            writer.writerow([patch_file, slice_num, patch_type, batch_idx, i, int(gt), int(pred), float(conf)])


    print(f"Saved prediction outputs to {csv_path}")

    # Metrics and plots
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    metrics_str = (f"Accuracy: {acc:.4f}\n"
                   f"Precision: {prec:.4f}\n"
                   f"Recall: {rec:.4f}\n"
                   f"F1: {f1:.4f}\n"
                   f"ROC-AUC: {roc_auc:.4f}\n"
                   f"Confusion matrix:\n{cm}")
    print(metrics_str)
    with open(os.path.join(qc_dir, "metrics.txt"), "w") as f:
        f.write(metrics_str)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "roc_curve.png"))
    plt.close()

    plt.figure()
    plt.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Saved metrics and curves to {qc_dir}")

    # ------------------------------------------------
    # IMAGE SAVING CODE (currently commented out)
    # ------------------------------------------------

    # def plot_patch_with_masks(img, mask, outpath, meta=None, label=None, pred=None, prob=None):
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(img, cmap='gray', interpolation='none')
    #     legend_handles = []
    #     mask_kidney = (mask == 1)
    #     mask_tumor  = (mask == 2)
    #     mask_cyst   = (mask == 3)
    #     if mask_kidney.any():
    #         plt.imshow(mask_kidney, alpha=0.5, cmap="Blues", interpolation='none')
    #         legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Blues")(0.6)))
    #     if mask_tumor.any():
    #         plt.imshow(mask_tumor, alpha=0.5, cmap="Reds", interpolation='none')
    #         legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Reds")(0.6)))
    #     if mask_cyst.any():
    #         plt.imshow(mask_cyst, alpha=0.5, cmap="Oranges", interpolation='none')
    #         legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Oranges")(0.6)))
    #     label_names = []
    #     if mask_kidney.any(): label_names.append("Kidney")
    #     if mask_tumor.any():  label_names.append("Tumor")
    #     if mask_cyst.any():   label_names.append("Cyst")
    #     if legend_handles:
    #         plt.legend(legend_handles, label_names, fontsize=8, loc='lower right')
    #     meta_str = "?"
    #     type_str = "?"
    #     if isinstance(meta, dict):
    #         meta_str = meta.get('slice', '?')
    #         type_str = meta.get('type', '?')
    #     elif isinstance(meta, str):
    #         if meta.startswith("{") and meta.endswith("}"):
    #             try:
    #                 import ast
    #                 meta_dict = ast.literal_eval(meta)
    #                 meta_str = meta_dict.get('slice', '?')
    #                 type_str = meta_dict.get('type', '?')
    #             except Exception:
    #                 pass
    #         else:
    #             type_str = meta
    #     title = f"Slice {meta_str}, Type: {type_str}"
    #     if label is not None and prob is not None and pred is not None:
    #         title += f"\nGT: {label}, Pred: {pred}, Prob: {prob:.2f}"
    #     plt.title(title, fontsize=8)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(outpath)
    #     plt.close()

    # idx_conf_pos = np.argsort(-all_probs)[:args.num_examples]
    # idx_conf_neg = np.argsort(all_probs)[:args.num_examples]
    # idx_uncertain = np.argsort(np.abs(all_probs - 0.5))[:args.num_examples]

    # def save_examples(indices, name):
    #     for i, idx in enumerate(indices):
    #         if idx >= len(all_imgs): continue
    #         img = all_imgs[idx]
    #         mask = all_masks[idx] if idx < len(all_masks) else np.zeros_like(img)
    #         meta = all_meta[idx] if idx < len(all_meta) else {}
    #         outpath = os.path.join(qc_dir, f"{name}_{i}_label{all_labels[idx]}_pred{all_preds[idx]}_prob{all_probs[idx]:.2f}.png")
    #         plot_patch_with_masks(img, mask, outpath, meta=meta, label=all_labels[idx], pred=all_preds[idx], prob=all_probs[idx])

    # save_examples(idx_conf_pos, "confident_lesion")
    # save_examples(idx_conf_neg, "confident_nonlesion")
    # save_examples(idx_uncertain, "uncertain")

    # print(f"Saved metrics, curves, and {args.num_examples*3} example patch images to {qc_dir}")

if __name__ == "__main__":
    main()
