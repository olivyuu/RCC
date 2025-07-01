import os
import torch
import numpy as np
from dev.detection.dataset import RCCPatchDataset
from dev.detection.model import get_model
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CyclicLR


def save_log(logfile, message):
    with open(logfile, 'a') as f:
        f.write(message + '\n')

def train_detection(config, run_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logfile = os.path.join(run_dir, 'train.log')

    # --- Dataset config ---
    data_conf = config.get('data', {})
    data_dir = data_conf.get('processed_dir', 'dev/data/processed/kits23/')
    batch_size = data_conf.get('batch_size', 32)
    num_workers = data_conf.get('num_workers', 2)
    split_seed = data_conf.get('split_seed', 42)
    split_frac = data_conf.get('train_frac', 0.8)
    augment = data_conf.get('augment', True)

    # --- Training config ---
    train_conf = config.get('train', {})
    epochs = train_conf.get('epochs', 50)
    lr = train_conf.get('lr', 1e-4)
    weight_decay = train_conf.get('weight_decay', 0.0)
    optimizer_name = train_conf.get('optimizer', 'adam').lower()
    # Scheduler support (optional)
    scheduler_type = train_conf.get('scheduler', 'step')
    scheduler_step_size = train_conf.get('scheduler_step_size', 40)
    scheduler_gamma = train_conf.get('scheduler_gamma', 0.5)
    base_lr = train_conf.get('base_lr', 1e-4)
    max_lr = train_conf.get('max_lr', 1e-3)
    step_size_up = train_conf.get('step_size_up', 10)

    # --- Model config ---
    model_conf = config.get('model', {})
    arch = model_conf.get('arch', "densenet121")
    pretrained = model_conf.get('pretrained', False)

    # --- Datasets ---
    train_set = RCCPatchDataset(data_dir, split='train', split_seed=split_seed, split_frac=split_frac, augment=augment)
    val_set = RCCPatchDataset(data_dir, split='val', split_seed=split_seed, split_frac=split_frac, augment=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Batches per epoch:", len(train_loader)) #help set step_size_up
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model ---
    model = get_model(pretrained=pretrained).to(device)

    # --- Optimizer and Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Scheduler ---
    if scheduler_type == 'cyclic':
        scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode='triangular2',
            cycle_momentum=False
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    else:
        scheduler = None


    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_preds, train_labels = [], []

        for images, labels, _, metas in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_type == 'cyclic':
                scheduler.step()
            train_loss.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy() > 0.5)
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_loss_mean = np.mean(train_loss)

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for images, labels, _, metas in val_loader:
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

        # Step scheduler if present
        if scheduler is not None and scheduler_type != 'cyclic':
            scheduler.step()

    # Final summary
    save_log(logfile, f"Best epoch: {best_epoch} (val_acc {best_val_acc:.4f})")
    print(f"Best epoch: {best_epoch} (val_acc {best_val_acc:.4f})")
