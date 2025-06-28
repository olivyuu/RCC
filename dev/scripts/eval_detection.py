import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from dev.detection.model import get_model
from dev.detection.dataset import RCCPatchDataset

def plot_patch_with_masks(img, mask, outpath, meta=None):
    """Overlay kidney/tumor/cyst masks if present, else show grayscale image."""
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    legend_handles = []
    mask_kidney = (mask == 1)
    mask_tumor  = (mask == 2)
    mask_cyst   = (mask == 3)
    # Overlay each mask with different color (kidney=blue, tumor=red, cyst=orange)
    if mask_kidney.any():
        plt.imshow(np.ma.masked_where(mask_kidney == 0, mask_kidney), alpha=0.3, cmap="Blues")
        legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Blues")(0.6)))
    if mask_tumor.any():
        plt.imshow(np.ma.masked_where(mask_tumor == 0, mask_tumor), alpha=0.3, cmap="Reds")
        legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Reds")(0.6)))
    if mask_cyst.any():
        plt.imshow(np.ma.masked_where(mask_cyst == 0, mask_cyst), alpha=0.3, cmap="Oranges")
        legend_handles.append(plt.Rectangle((0,0),1,1, color=plt.get_cmap("Oranges")(0.6)))
    label_names = []
    if mask_kidney.any(): label_names.append("Kidney")
    if mask_tumor.any(): label_names.append("Tumor")
    if mask_cyst.any():  label_names.append("Cyst")
    if legend_handles:
        plt.legend(legend_handles, label_names, fontsize=8, loc='lower right')
    plt.axis('off')
    # Fix: robustly handle meta as dict or str
    if meta is not None:
        if isinstance(meta, dict):
            plt.title(f"Slice {meta.get('slice', '?')}, Type: {meta.get('type', '?')}", fontsize=8)
        else:
            plt.title(str(meta), fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--processed_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_examples', type=int, default=10)
    args = parser.parse_args()

    # Output QC dir
    qc_dir = os.path.join(args.run_dir, 'qc')
    os.makedirs(qc_dir, exist_ok=True)

    # ---- Load data ----
    print(f"Loading {args.split} dataset from {args.processed_dir} ...")
    dataset = RCCPatchDataset(args.processed_dir, split=args.split, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # ---- Load model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ---- Inference ----
    all_preds = []
    all_labels = []
    all_meta = []
    all_probs = []
    all_imgs = []
    all_masks = []

    with torch.no_grad():
        for images, labels, metas in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_meta.extend(metas)
            # Also store images/masks for visualization
            imgs_np = images.cpu().numpy()
            for i in range(imgs_np.shape[0]):
                # RCCPatchDataset guarantees 1 channel, HxW
                img = imgs_np[i,0] if imgs_np[i].ndim == 3 else imgs_np[i]
                all_imgs.append(img)
            # Try to get mask from meta (store if possible)
            if hasattr(dataset, 'masks'):
                masks_np = dataset.masks
                batch_indices = list(range(len(all_masks), len(all_masks) + len(labels)))
                for i, idx in enumerate(batch_indices):
                    if idx < len(masks_np):
                        mask = masks_np[idx][:,:,0]
                        all_masks.append(mask)
                    else:
                        all_masks.append(np.zeros_like(all_imgs[0]))  # fallback
            else:
                all_masks.extend([np.zeros_like(all_imgs[0])] * len(labels))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # ---- Metrics ----
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

    # ---- ROC Curve ----
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

    # ---- Confusion matrix ----
    plt.figure()
    plt.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "confusion_matrix.png"))
    plt.close()

    # ---- Example images ----
    idx_conf_pos = np.argsort(-all_probs)[:args.num_examples]
    idx_conf_neg = np.argsort(all_probs)[:args.num_examples]
    idx_uncertain = np.argsort(np.abs(all_probs - 0.5))[:args.num_examples]

    def save_examples(indices, name):
        for i, idx in enumerate(indices):
            if idx >= len(all_imgs): continue
            img = all_imgs[idx]
            mask = all_masks[idx] if idx < len(all_masks) else np.zeros_like(img)
            meta = all_meta[idx] if idx < len(all_meta) else {}
            outpath = os.path.join(qc_dir, f"{name}_{i}_label{all_labels[idx]}_prob{all_probs[idx]:.2f}.png")
            plot_patch_with_masks(img, mask, outpath, meta=meta)

    save_examples(idx_conf_pos, "confident_lesion")
    save_examples(idx_conf_neg, "confident_nonlesion")
    save_examples(idx_uncertain, "uncertain")

    print(f"Saved metrics, curves, and {args.num_examples*3} example patch images to {qc_dir}")

if __name__ == "__main__":
    main()
