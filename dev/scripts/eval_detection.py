import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)
from torchvision.utils import save_image

from dev.detection.model import get_model  # assumes your build_model loads arch from config
from dev.detection.dataset import RCCPatchDataset

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return roc_auc

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ['Background', 'Lesion']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def overlay_mask(image, mask, mask_color=(255,0,0), alpha=0.4):
    """Overlay a binary mask on grayscale image (for QC visualization)."""
    img = np.stack([image]*3, axis=-1)  # Grayscale to RGB
    mask_rgb = np.zeros_like(img)
    for c in range(3): mask_rgb[:,:,c] = mask_color[c]
    mask_bool = (mask > 0)
    img = img * (1-alpha) + mask_rgb * alpha * mask_bool[:,:,None]
    img = np.clip(img, 0, 1)
    return img

def save_patch_with_mask(image, mask, pred_prob, gt_label, out_path, title=None):
    img_disp = image.squeeze()
    mask_disp = mask.squeeze()
    overlay = overlay_mask(img_disp, mask_disp, mask_color=(255,0,0), alpha=0.4)
    plt.figure(figsize=(3, 3))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(title or f"GT: {gt_label} | Pred: {pred_prob:.2f}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--run_dir', type=str, required=True, help='Base directory for this run (for saving QC)')
    parser.add_argument('--processed_dir', type=str, required=True, help='Processed patch dataset dir')
    parser.add_argument('--split', type=str, default='val', help='Which split to use (val or test)')
    parser.add_argument('--num_examples', type=int, default=8, help='How many confident/uncertain examples to save')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qc_dir = os.path.join(args.run_dir, 'qc')
    os.makedirs(qc_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(args.run_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset
    print(f"Loading {args.split} dataset from {args.processed_dir} ...")
    val_dataset = RCCPatchDataset(
        args.processed_dir,
        split=args.split,
        split_seed=config['data']['split_seed'],
        split_frac=config['data']['train_frac'],
        augment=False
    )
    print(f"Loaded {len(val_dataset)} patches for evaluation.")

    # Load model
    # Load model
    model = get_model(config['model'].get('pretrained', False))
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()


    all_probs, all_labels, all_imgs, all_masks, all_meta = [], [], [], [], []
    batch_size = 64
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Collect predictions
    with torch.no_grad():
        for imgs, labels, metas in loader:
            imgs = imgs.to(device)  # [B,1,224,224]
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            # If 2-class softmax, take class 1
            if len(probs.shape) > 1 and probs.shape[1] == 2:
                probs = probs[:,1]
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            # For visualization
            all_imgs.append(imgs.cpu().numpy())
            # Get mask (from meta dict, requires loading .npz file for each case/patch)
            # We'll just get mask from meta for now
            all_masks.extend([(meta['mask'] if isinstance(meta, dict) and 'mask' in meta else np.zeros((224,224))) for meta in metas])
            all_meta.extend(metas)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_imgs = np.concatenate(all_imgs, axis=0)

    # Compute metrics
    y_pred = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, y_pred)
    prec = precision_score(all_labels, y_pred)
    rec = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)
    cm = confusion_matrix(all_labels, y_pred)
    roc_auc = plot_roc(all_labels, all_probs, os.path.join(qc_dir, "roc_curve.png"))
    plot_confusion(all_labels, y_pred, os.path.join(qc_dir, "confusion_matrix.png"))

    metrics_str = (
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1: {f1:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}\n"
        f"Confusion matrix:\n{cm}\n"
    )
    print(metrics_str)
    with open(os.path.join(qc_dir, "metrics.txt"), 'w') as f:
        f.write(metrics_str)

    # Failure analysis
    idx_conf_pos = np.argsort(-all_probs)[:args.num_examples]
    idx_conf_neg = np.argsort(all_probs)[:args.num_examples]
    idx_uncertain = np.argsort(np.abs(all_probs - 0.5))[:args.num_examples]

    def save_examples(idxs, name):
        for i, idx in enumerate(idxs):
            img = all_imgs[idx].squeeze()
            label = all_labels[idx]
            prob = all_probs[idx]
            meta = all_meta[idx]
            # Overlay mask: only for positive (if available)
            mask = np.zeros((224,224))
            if isinstance(meta, dict) and 'mask' in meta:
                mask = meta['mask']
            elif hasattr(meta, 'get') and meta.get('mask') is not None:
                mask = meta['mask']
            save_patch_with_mask(img, mask, prob, label, os.path.join(qc_dir, f"{name}_{i}_label{label}_prob{prob:.2f}.png"),
                                 title=f"{name}: prob={prob:.2f}, label={label}")
    save_examples(idx_conf_pos, "confident_lesion")
    save_examples(idx_conf_neg, "confident_bg")
    save_examples(idx_uncertain, "uncertain")

    # False positive/negatives
    fp_idxs = np.where((y_pred == 1) & (all_labels == 0))[0][:args.num_examples]
    fn_idxs = np.where((y_pred == 0) & (all_labels == 1))[0][:args.num_examples]
    save_examples(fp_idxs, "false_positive")
    save_examples(fn_idxs, "false_negative")

    print(f"QC/analysis images and metrics saved to: {qc_dir}")

if __name__ == "__main__":
    main()
