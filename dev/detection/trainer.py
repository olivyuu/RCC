# dev/detection/trainer.py

import os
import torch
import numpy as np
from dev.detection.dataset import RCCPatchDataset
from dev.detection.model import get_model
from sklearn.metrics import accuracy_score

def save_log(logfile, message):
    with open(logfile, 'a') as f:
        f.write(message + '\n')

def train_detection(config, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logfile = os.path.join(run_dir, 'train.log')

    # Dataset
    data_dir = config.get('data', {}).get('processed_dir', 'dev/data/processed/kits23/')
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 50)
    lr = config.get('lr', 1e-4)
    split_seed = config.get('split_seed', 42)
    split_frac = config.get('split_frac', 0.8)
    augment = config.get('augment', True)

    train_set = RCCPatchDataset(data_dir, split='train', split_seed=split_seed, split_frac=split_frac, augment=augment)
    val_set = RCCPatchDataset(data_dir, split='val', split_seed=split_seed, split_frac=split_frac, augment=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model/optimizer
    model = get_model(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_preds, train_labels = [], []

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_loss_mean = np.mean(train_loss)

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                val_loss.append(loss.item())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy() > 0.5)
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_loss_mean = np.mean(val_loss)

        line = (f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss_mean:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss_mean:.4f}, Acc: {val_acc:.4f}")
        print(line)
        save_log(logfile, line)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch+1
            best_model_path = os.path.join(run_dir, 'best_model.pt')
            torch.save({'model': model.state_dict(), 'config': config}, best_model_path)
            save_log(logfile, f"Best model updated (epoch {best_epoch}, val_acc {best_val_acc:.4f})")

    # Final summary
    save_log(logfile, f"Best epoch: {best_epoch} (val_acc {best_val_acc:.4f})")
    print(f"Best epoch: {best_epoch} (val_acc {best_val_acc:.4f})")
