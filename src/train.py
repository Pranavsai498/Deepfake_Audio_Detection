import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import HybridCNNLSTMDetector
from src.dataset import AudioDeepfakeDataset
from src.evaluation.eer import compute_eer
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def set_seed(seed: int=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_fake_probs = ([], [], [])
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feats, labels, _, _, _ in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_fake_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    all_fake_probs = np.array(all_fake_probs)
    all_preds = np.array(all_preds)
    avg_loss = total_loss / max(1, len(all_labels))
    acc = accuracy_score(all_labels, all_preds) * 100.0
    try:
        eer_res = compute_eer(all_labels, all_fake_probs)
        eer = float(eer_res['eer']) * 100.0
        optimal_thresh = float(eer_res['threshold'])
    except Exception:
        eer = 50.0
        optimal_thresh = 0.5
    return (avg_loss, acc, eer, optimal_thresh, all_labels, all_fake_probs)

def train(train_csv: str='data/metadata/train.csv', val_csv: str='data/metadata/val.csv', epochs: int=15, batch_size: int=32, lr: float=0.0007, weight_decay: float=0.0001, patience: int=5, checkpoint_dir: str='checkpoints', results_dir: str='results', resume_checkpoint: str=None):
    set_seed(42)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n========================================================')
    print(f'  Training Hybrid CNN-BiLSTM Deepfake Audio Detector')
    print(f'  Execution Device: {device} | Python: {sys.version.split()[0]}')
    print(f'========================================================')
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"Train Dataset: {len(train_df)} clips. Class distribution: {dict(train_df['label'].value_counts())}")
    print(f"Val Dataset:   {len(val_df)} clips. Class distribution: {dict(val_df['label'].value_counts())}")
    train_dataset = AudioDeepfakeDataset(train_df, is_train=True, target_frames=300)
    val_dataset = AudioDeepfakeDataset(val_df, is_train=False, target_frames=300)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = HybridCNNLSTMDetector().to(device)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f'Loading pretrained weights from {resume_checkpoint} for warm-start training...')
        try:
            ckpt = torch.load(resume_checkpoint, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            print('Successfully loaded existing checkpoint weights.')
        except Exception as e:
            print(f'Warning: Could not load checkpoint ({e}), initializing fresh weights.')
    print(f'Trainable Parameters: {model.count_parameters():,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_val_loss = float('inf')
    best_val_eer = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_detector.pt')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_eer': [], 'lr': []}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_samples = 0
        for feats, labels, _, _, _ in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            n_samples += len(labels)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = train_loss / max(1, n_samples)
        val_loss, val_acc, val_eer, val_tau, _, _ = evaluate_split(model, val_loader, device)
        elapsed = time.time() - t0
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_eer'].append(val_eer)
        history['lr'].append(cur_lr)
        print(f'Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val EER: {val_eer:.2f}% (tau={val_tau:.4f}) | LR: {cur_lr:.6f}')
        if val_eer < best_val_eer or (abs(val_eer - best_val_eer) < 0.2 and val_loss < best_val_loss):
            best_val_eer = val_eer
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'val_eer': val_eer, 'optimal_tau': val_tau, 'val_loss': val_loss, 'history': history, 'architecture': 'HybridCNNLSTMDetector'}, best_checkpoint_path)
            print(f'  --> Saved new best detector checkpoint to {best_checkpoint_path} (Val EER: {val_eer:.2f}%, Optimal Tau: {val_tau:.4f})')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch} epochs (Patience = {patience}).')
                break
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Cross-Entropy Loss vs. Epochs', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['val_eer']) + 1), history['val_eer'], 'g-^', label='Val EER (%)')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], 'm--d', label='Val Acc (%)')
    plt.title('Validation EER (%) & Accuracy (%)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=200)
    plt.close()
    print(f"Saved loss & convergence plots to {os.path.join(results_dir, 'training_loss.png')}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--resume', type=str, default='checkpoints/best_detector_v1_backup.pt')
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, resume_checkpoint=args.resume)
