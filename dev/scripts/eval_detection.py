# dev/scripts/eval_detection.py

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from dev.detection.dataset import RCCPatchDataset
from dev.detection.model import get_model

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    # Load config/model/checkpoint
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    dataset = RCCPatchDataset(config["data"]["processed_dir"], split='val', augment=False)
    X = torch.tensor(dataset.images).float().unsqueeze(1).to(device) # (N, 1, H, W)
    y = torch.tensor(dataset.labels).float().to(device)
    metas = dataset.metas

    with torch.no_grad():
        logits = model(X).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

    # Metrics
    acc = (preds == y.long()).float().mean().item()
    print(f"Model: {args.ckpt}\nConfig: {args.config}")
    print(f"Accuracy: {acc:.4f}")

    # Confusion Matrix and ROC
    cm = confusion_matrix(y.cpu().numpy(), preds.cpu().numpy())
    print(f"Confusion Matrix:\n{cm}")
    auc = roc_auc_score(y.cpu().numpy(), probs.cpu().numpy())
    print(f"ROC-AUC: {auc:.4f}")

    # Output images: confident and uncertain
    conf_thresh = 0.9
    uncertain_thresh = 0.55  # Near 0.5
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    confident_idxs = ((probs > conf_thresh) | (probs < 1-conf_thresh)).nonzero().flatten()
    uncertain_idxs = ((probs > (1-uncertain_thresh)) & (probs < uncertain_thresh)).nonzero().flatten()

    for idx in confident_idxs[:5]:
        img = X[idx,0].cpu().numpy()
        pred = preds[idx].item()
        p = probs[idx].item()
        label = y[idx].item()
        plt.imshow(img, cmap='gray')
        plt.title(f"Confident | P={p:.2f} | Pred={pred} | True={label}")
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"confident_{idx}_P{p:.2f}_T{label}.png"))
        plt.close()

    for idx in uncertain_idxs[:5]:
        img = X[idx,0].cpu().numpy()
        pred = preds[idx].item()
        p = probs[idx].item()
        label = y[idx].item()
        plt.imshow(img, cmap='gray')
        plt.title(f"Uncertain | P={p:.2f} | Pred={pred} | True={label}")
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"uncertain_{idx}_P{p:.2f}_T{label}.png"))
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='dev/runs/eval/')
    args = parser.parse_args()
    main(args)
