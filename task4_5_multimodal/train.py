"""
=========================================================
 Multimodal Training Script (Task 4 & 5) — v2 SIMPLIFIED
 
 Key changes from v1:
   - SAME architecture as Task 1/2 (LSTM 32, not BiLSTM 64)
   - Transfer learning from Task 1/2 pretrained weights
   - Regression-based multi-class (same approach as Task 1/2)
   - Better training: no drop_last, patience=25, LR=0.0005
   - Ablation report auto-generated
 
 Usage:
   python task4_5_multimodal/train.py \
     --visual_feature_dir dataset/features \
     --rppg_feature_dir data/rppg_features \
     --labels dataset/train/labels_train.xlsx \
     --output_dir task4_5_multimodal \
     --pretrained task1_2_seq/model_seq.pth
=========================================================
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalFusionModel, VisualOnlyModel

# ========================= CONFIG =========================
NUM_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 8
LR = 0.0005
SEQ_LEN = 100
VISUAL_DIM = 9
RPPG_DIM = 14
SEED = 42
PATIENCE = 25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_to_class(val):
    """Convert regression label to 4-class: 0.0→0, 0.33→1, 0.66→2, 1.0→3"""
    val_r = round(float(val), 2)
    if val_r <= 0.0: return 0
    if val_r <= 0.33: return 1
    if val_r <= 0.66: return 2
    return 3


def label_to_binary(val):
    return 1.0 if float(val) > 0.33 else 0.0


def regression_to_class(pred):
    """Convert regression prediction to class (same as Task 1/2)."""
    if pred < 0.165: return 0
    if pred < 0.5: return 1
    if pred < 0.835: return 2
    return 3


# ========================= DATASET =========================

class MultimodalDataset(Dataset):
    """Dataset that loads visual sequences + rPPG feature vectors."""
    
    def __init__(self, visual_paths, rppg_paths, labels, training=True, oversample=True):
        self.training = training
        self.data = []
        
        valid_data = []
        for vp, rp, label in zip(visual_paths, rppg_paths, labels):
            try:
                if not os.path.exists(vp):
                    continue
                visual = np.load(vp)
                if visual.shape[0] == 0:
                    continue
                if visual.shape[1] < VISUAL_DIM:
                    pad = np.zeros((visual.shape[0], VISUAL_DIM - visual.shape[1]))
                    visual = np.hstack([visual, pad])
                elif visual.shape[1] > VISUAL_DIM:
                    visual = visual[:, :VISUAL_DIM]
                
                # Load rPPG features (14,)
                if os.path.exists(rp):
                    rppg = np.load(rp)
                    if rppg.shape[0] < RPPG_DIM:
                        rppg = np.pad(rppg, (0, RPPG_DIM - rppg.shape[0]))
                    elif rppg.shape[0] > RPPG_DIM:
                        rppg = rppg[:RPPG_DIM]
                else:
                    rppg = np.zeros(RPPG_DIM, dtype=np.float32)
                
                # Replace NaN/Inf
                visual = np.nan_to_num(visual, nan=0.0, posinf=0.0, neginf=0.0)
                rppg = np.nan_to_num(rppg, nan=0.0, posinf=0.0, neginf=0.0)
                
                cls = label_to_class(label)
                valid_data.append((visual, rppg, label, cls))
            except Exception:
                continue
        
        # Oversampling for class balance
        if training and oversample and valid_data:
            class_counts = Counter([d[3] for d in valid_data])
            max_count = max(class_counts.values())
            
            oversampled = []
            for cls_id in sorted(class_counts.keys()):
                cls_samples = [d for d in valid_data if d[3] == cls_id]
                repeats = max(1, max_count // len(cls_samples))
                for _ in range(repeats):
                    oversampled.extend(cls_samples)
            self.data = oversampled
            print(f"    Oversampled: {len(valid_data)} → {len(self.data)}")
        else:
            self.data = valid_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        visual, rppg, label, cls = self.data[idx]
        
        # Normalize visual features per-sample (same as Task 1/2)
        mean = np.mean(visual, axis=0)
        std = np.std(visual, axis=0) + 1e-6
        visual = (visual - mean) / std
        
        # Augmentation (training only)
        if self.training:
            if random.random() < 0.3:
                visual = visual + np.random.randn(*visual.shape) * 0.02
            if random.random() < 0.2:
                rppg = rppg + np.random.randn(*rppg.shape) * 0.01
        
        # Normalize rPPG features
        rppg_norm = rppg.copy()
        for i in [0, 1, 2, 6]:
            if i < len(rppg_norm):
                rppg_norm[i] = rppg_norm[i] / 100.0
        if len(rppg_norm) > 7:
            rppg_norm[7] = rppg_norm[7] / 50.0
        for i in [8, 9, 10]:
            if i < len(rppg_norm):
                rppg_norm[i] = rppg_norm[i] * 10.0
        for i in [11, 12]:
            if i < len(rppg_norm):
                rppg_norm[i] = np.log1p(abs(rppg_norm[i]))
        if len(rppg_norm) > 13:
            rppg_norm[13] = min(rppg_norm[13] / 5.0, 1.0)
        
        # Pad/subsample visual to SEQ_LEN
        L, D = visual.shape
        if L < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - L, D))
            visual = np.vstack([visual, pad])
        elif L > SEQ_LEN:
            indices = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
            visual = visual[indices]
        
        visual_t = torch.FloatTensor(visual)
        rppg_t = torch.FloatTensor(rppg_norm)
        bin_target = torch.tensor(label_to_binary(label), dtype=torch.float32)
        reg_target = torch.tensor(float(label), dtype=torch.float32)
        
        return visual_t, rppg_t, bin_target, reg_target


# ========================= TRAINING =========================

def train_epoch(model, loader, optimizer, crit_bin, crit_reg, device, multimodal=True):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        if multimodal:
            visual, rppg, bin_tgt, reg_tgt = batch
            visual, rppg = visual.to(device), rppg.to(device)
        else:
            visual, _, bin_tgt, reg_tgt = batch
            visual = visual.to(device)
        
        bin_tgt = bin_tgt.to(device)
        reg_tgt = reg_tgt.to(device)
        
        optimizer.zero_grad()
        
        if multimodal:
            out_bin, out_multi, out_reg = model(visual, rppg)
        else:
            out_bin, out_multi, out_reg = model(visual)
        
        loss_b = crit_bin(out_bin.squeeze(-1), bin_tgt)
        
        # Regression loss for multi-class (same as Task 1/2) — used for BOTH models
        reg_pred = torch.sigmoid(out_reg.squeeze(-1))
        loss_r = crit_reg(reg_pred, reg_tgt)
        loss = loss_b + 2.0 * loss_r
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device, multimodal=True):
    model.eval()
    preds_bin, preds_cls = [], []
    actuals_bin, actuals_cls = [], []
    
    with torch.no_grad():
        for batch in loader:
            if multimodal:
                visual, rppg, bin_tgt, reg_tgt = batch
                visual, rppg = visual.to(device), rppg.to(device)
            else:
                visual, _, bin_tgt, reg_tgt = batch
                visual = visual.to(device)
            
            if multimodal:
                out_bin, out_multi, out_reg = model(visual, rppg)
            else:
                out_bin, out_multi, out_reg = model(visual)
            
            # Binary
            prob = torch.sigmoid(out_bin).squeeze(-1)
            preds_bin.extend((prob.cpu().numpy() > 0.5).astype(int).tolist())
            actuals_bin.extend(bin_tgt.numpy().astype(int).tolist())
            
            # Multi-class from regression (same approach as Task 1/2) — for BOTH models
            reg_pred = torch.sigmoid(out_reg).squeeze(-1).cpu().numpy()
            for rp in reg_pred:
                preds_cls.append(regression_to_class(float(rp)))
            
            for rt in reg_tgt.numpy():
                actuals_cls.append(label_to_class(float(rt)))
    
    acc_bin = accuracy_score(actuals_bin, preds_bin)
    f1_bin = f1_score(actuals_bin, preds_bin, average='binary', zero_division=0)
    acc_cls = accuracy_score(actuals_cls, preds_cls)
    f1_cls = f1_score(actuals_cls, preds_cls, average='macro', zero_division=0)
    
    return acc_bin, f1_bin, acc_cls, f1_cls


def train_model(args, visual_paths, rppg_paths, labels, classes, 
                model_type='multimodal', output_prefix='fusion'):
    """Train either multimodal or visual-only model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    best_overall_acc = 0.0
    best_fold = -1
    
    is_multi = (model_type == 'multimodal')
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(visual_paths, classes)):
        print(f"\n  FOLD {fold+1}/{NUM_FOLDS} ({model_type})")
        
        train_vp = [visual_paths[i] for i in train_idx]
        train_rp = [rppg_paths[i] for i in train_idx]
        train_y = [labels[i] for i in train_idx]
        val_vp = [visual_paths[i] for i in val_idx]
        val_rp = [rppg_paths[i] for i in val_idx]
        val_y = [labels[i] for i in val_idx]
        
        train_ds = MultimodalDataset(train_vp, train_rp, train_y, training=True)
        val_ds = MultimodalDataset(val_vp, val_rp, val_y, training=False, oversample=False)
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        
        if is_multi:
            model = MultimodalFusionModel(
                visual_dim=VISUAL_DIM, rppg_dim=RPPG_DIM,
                lstm_hidden=32, lstm_layers=2, dropout=0.3
            ).to(device)
            # Transfer learning from Task 1/2
            if args.pretrained and os.path.exists(args.pretrained):
                model.load_task12_weights(args.pretrained)
        else:
            model = VisualOnlyModel(
                visual_dim=VISUAL_DIM, hidden_dim=32, 
                num_layers=2, dropout=0.3
            ).to(device)
            # Transfer learning from Task 1/2
            if args.pretrained and os.path.exists(args.pretrained):
                model.load_task12_weights(args.pretrained)
        
        crit_bin = nn.BCEWithLogitsLoss()
        crit_reg = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_metric = 0.0
        best_bin_fold = 0.0
        best_cls_fold = 0.0
        patience = 0
        
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, crit_bin, crit_reg, device, is_multi)
            scheduler.step()
            
            acc_bin, f1_bin, acc_cls, f1_cls = evaluate(model, val_loader, device, is_multi)
            
            # Track best by combined metric (binary more important)
            metric = acc_bin * 0.6 + acc_cls * 0.4
            
            improved = False
            if metric > best_metric:
                best_metric = metric
                best_bin_fold = acc_bin
                best_cls_fold = acc_cls
                patience = 0
                improved = True
                fold_path = os.path.join(args.output_dir, f'{output_prefix}_fold{fold}.pth')
                torch.save(model.state_dict(), fold_path)
            else:
                patience += 1
            
            if epoch % 10 == 0 or improved:
                marker = " ★" if improved else ""
                print(f"    Ep {epoch:3d}: Loss={loss:.3f} | Bin={acc_bin*100:.1f}% "
                      f"Multi={acc_cls*100:.1f}% F1={f1_cls:.3f}{marker}")
            
            if patience >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        print(f"  Fold {fold+1}: Bin={best_bin_fold*100:.1f}% | Multi={best_cls_fold*100:.1f}%")
        fold_results.append({'fold': fold, 'acc_bin': best_bin_fold, 'acc_cls': best_cls_fold})
        
        if best_metric > best_overall_acc:
            best_overall_acc = best_metric
            best_fold = fold
    
    # Copy best fold
    if best_fold >= 0:
        import shutil
        src = os.path.join(args.output_dir, f'{output_prefix}_fold{best_fold}.pth')
        dst = os.path.join(args.output_dir, f'{output_prefix}_best.pth')
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Task 4/5 Multimodal Training")
    parser.add_argument('--visual_feature_dir', type=str, required=True,
                        help='Dir with visual .npy features (from Task 1/2)')
    parser.add_argument('--rppg_feature_dir', type=str, required=True,
                        help='Dir with rPPG .npy features (from extract_rppg_features.py)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels Excel')
    parser.add_argument('--pretrained', type=str, default='task1_2_seq/model_seq.pth',
                        help='Path to pretrained Task 1/2 model for transfer learning')
    parser.add_argument('--output_dir', type=str, default='task4_5_multimodal',
                        help='Where to save models')
    args = parser.parse_args()
    
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load labels
    df = pd.read_excel(args.labels)
    names = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].tolist()
    
    visual_paths = [os.path.join(args.visual_feature_dir, os.path.splitext(n)[0] + '.npy')
                    for n in names]
    rppg_paths = [os.path.join(args.rppg_feature_dir, os.path.splitext(n)[0] + '.npy')
                  for n in names]
    
    # Filter valid (need at least visual features)
    valid_vp, valid_rp, valid_labels = [], [], []
    for vp, rp, l in zip(visual_paths, rppg_paths, labels):
        if os.path.exists(vp):
            valid_vp.append(vp)
            valid_rp.append(rp)
            valid_labels.append(l)
    
    valid_classes = [label_to_class(l) for l in valid_labels]
    
    print(f"Dataset: {len(valid_vp)} samples")
    print(f"Class distribution: {dict(Counter(valid_classes))}")
    print(f"rPPG features available: {sum(1 for rp in valid_rp if os.path.exists(rp))}/{len(valid_rp)}")
    
    # ================================================================
    # STEP 1: Train Visual-Only Baseline (for ablation)
    # ================================================================
    print(f"\n{'='*60}")
    print(f" ABLATION: Training Visual-Only Baseline")
    print(f"{'='*60}")
    
    visual_results = train_model(args, valid_vp, valid_rp, valid_labels, valid_classes,
                                  model_type='visual_only', output_prefix='visual_only')
    
    # ================================================================
    # STEP 2: Train Multimodal Fusion Model
    # ================================================================
    print(f"\n{'='*60}")
    print(f" MAIN: Training Multimodal Fusion Model")
    print(f"{'='*60}")
    
    fusion_results = train_model(args, valid_vp, valid_rp, valid_labels, valid_classes,
                                  model_type='multimodal', output_prefix='fusion')
    
    # ================================================================
    # ABLATION REPORT
    # ================================================================
    print(f"\n{'='*60}")
    print(f" ABLATION REPORT: Visual-Only vs Multimodal")
    print(f"{'='*60}")
    
    if visual_results and fusion_results:
        v_bin = np.mean([r['acc_bin'] for r in visual_results]) * 100
        v_cls = np.mean([r['acc_cls'] for r in visual_results]) * 100
        f_bin = np.mean([r['acc_bin'] for r in fusion_results]) * 100
        f_cls = np.mean([r['acc_cls'] for r in fusion_results]) * 100
        
        print(f"\n  {'Model':<25} {'Binary':>10} {'Multi-class':>12}")
        print(f"  {'-'*50}")
        print(f"  {'Visual Only':<25} {v_bin:>9.1f}% {v_cls:>11.1f}%")
        print(f"  {'Visual + rPPG (Fusion)':<25} {f_bin:>9.1f}% {f_cls:>11.1f}%")
        print(f"  {'-'*50}")
        print(f"  {'Improvement':<25} {f_bin - v_bin:>+9.1f}% {f_cls - v_cls:>+11.1f}%")
        
        if f_bin > v_bin or f_cls > v_cls:
            print(f"\n  ✅ Multimodal outperforms Visual-Only!")
        else:
            print(f"\n  ⚠️  Multimodal did not outperform — check fusion/rPPG quality")
        
        # Save ablation report
        report_path = os.path.join(args.output_dir, 'ablation_report.txt')
        with open(report_path, 'w') as f:
            f.write("ABLATION REPORT: Visual-Only vs Multimodal\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Visual Only - Binary:      {v_bin:.1f}%\n")
            f.write(f"Visual Only - Multi-class:  {v_cls:.1f}%\n")
            f.write(f"Multimodal  - Binary:      {f_bin:.1f}%\n")
            f.write(f"Multimodal  - Multi-class:  {f_cls:.1f}%\n\n")
            f.write(f"Binary Improvement:      {f_bin - v_bin:+.1f}%\n")
            f.write(f"Multi-class Improvement:  {f_cls - v_cls:+.1f}%\n")
        print(f"\n  Ablation report saved: {report_path}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
