import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from model_lstm import EngagementLSTM
from tqdm import tqdm

# Config
NUM_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 8
LR = 0.001
SEQ_LEN = 100 # Fixed length for batching (pad/truncate)

class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels, training=True):
        self.data = []
        for path, label in zip(feature_paths, labels):
            if os.path.exists(path):
                try:
                    fsize = os.path.getsize(path)
                    if fsize < 100:  # Skip corrupted/empty files
                        continue
                    arr = np.load(path)
                    if arr.shape[0] == 0 or arr.shape[1] != 9:
                        continue
                    # Normalize per-sample
                    arr = (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + 1e-6)
                    self.data.append((arr, label))
                except Exception:
                    continue  # Skip corrupted files
        self.training = training

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feat, label = self.data[idx]
        
        # Pad or Truncate to SEQ_LEN
        L, D = feat.shape
        if L < SEQ_LEN:
            # Pad
            pad = np.zeros((SEQ_LEN - L, D))
            feat = np.vstack([feat, pad])
        elif L > SEQ_LEN:
            # Truncate / Random Sample if training? 
            # Sequential subsample is better for LSTM
            indices = np.linspace(0, L-1, SEQ_LEN, dtype=int)
            feat = feat[indices]
            
        feat = torch.FloatTensor(feat)
        
        # Labels
        # Binary: 0 (0, 0.33) | 1 (0.66, 1)
        # Regression: 0.0, 0.33, 0.66, 1.0 (Direct)
        
        # Convert label to float for Regression
        reg_target = torch.tensor(label, dtype=torch.float32)
        
        # Binary Target
        bin_target = torch.tensor(1.0 if label > 0.33 else 0.0, dtype=torch.float32)
        
        return feat, bin_target, reg_target

def bin_predictions(preds):
    # preds: array of floats
    # 0, 0.33, 0.66, 1.0
    # Thresholds: 0.165, 0.5, 0.835
    classes = []
    for p in preds:
        if p < 0.165: c = 0
        elif p < 0.5: c = 1 # 0.33
        elif p < 0.835: c = 2 # 0.66
        else: c = 3 # 1.0
        classes.append(c)
    return classes

def get_class_from_label(val):
    val = round(float(val), 2)
    if val == 0.0: return 0
    if val == 0.33: return 1
    if val == 0.66: return 2
    if val == 1.0: return 3
    return 0

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Labels
    df = pd.read_excel(args.labels)
    names = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].tolist()
    
    paths = [os.path.join(args.feature_dir, os.path.splitext(n)[0] + '.npy') for n in names]
    
    # Check valid
    valid_paths, valid_labels = [], []
    for p, l in zip(paths, labels):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_labels.append(l)
    
    print(f"Dataset: {len(valid_paths)} samples available.")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    fold_results_t1 = []
    
    # Task 2 metrics
    fold_acc_t2 = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(valid_paths)):
        print(f"\n=== FOLD {fold+1}/{NUM_FOLDS} ===")
        
        train_X = [valid_paths[i] for i in train_idx]
        train_y = [valid_labels[i] for i in train_idx]
        val_X = [valid_paths[i] for i in val_idx]
        val_y = [valid_labels[i] for i in val_idx]
        
        train_ds = FeatureDataset(train_X, train_y, training=True)
        val_ds = FeatureDataset(val_X, val_y, training=False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        model = EngagementLSTM().to(device)
        
        # Loss
        crit_bin = nn.BCEWithLogitsLoss()
        crit_reg = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        best_acc_t1 = 0.0
        best_acc_t2 = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            bin_losses = []
            reg_losses = []
            
            for feat, bin_tgt, reg_tgt in train_loader:
                feat, bin_tgt, reg_tgt = feat.to(device), bin_tgt.to(device), reg_tgt.to(device)
                
                optimizer.zero_grad()
                out_bin, out_reg = model(feat)
                
                loss_b = crit_bin(out_bin.squeeze(), bin_tgt)
                loss_r = crit_reg(out_reg.squeeze(), reg_tgt)
                
                # Multi-task loss
                loss = loss_b + loss_r
                loss.backward()
                optimizer.step()
                
                bin_losses.append(loss_b.item())
                reg_losses.append(loss_r.item())
        
            # Validation
            model.eval()
            actuals_bin = []
            preds_bin = []
            
            actuals_cls = [] # 0,1,2,3
            preds_reg = [] # raw float
            
            with torch.no_grad():
                for feat, bin_tgt, reg_tgt in val_loader:
                    feat = feat.to(device)
                    out_bin, out_reg = model(feat)
                    
                    # Task 1
                    prob = torch.sigmoid(out_bin).item()
                    preds_bin.append(1 if prob > 0.5 else 0)
                    actuals_bin.append(int(bin_tgt.item()))
                    
                    # Task 2
                    preds_reg.append(out_reg.item())
                    actuals_cls.append(get_class_from_label(reg_tgt.item()))
            
            # Metrics
            acc_t1 = accuracy_score(actuals_bin, preds_bin)
            
            # Regression -> Class
            pred_classes = bin_predictions(preds_reg)
            acc_t2 = accuracy_score(actuals_cls, pred_classes)
            
            if acc_t1 > best_acc_t1: best_acc_t1 = acc_t1
            if acc_t2 > best_acc_t2: best_acc_t2 = acc_t2
            
            if epoch % 10 == 0:
                print(f"  Ep {epoch}: Loss B={np.mean(bin_losses):.3f} R={np.mean(reg_losses):.3f} | T1 Acc={acc_t1:.2f} | T2 Acc={acc_t2:.2f}")

        print(f"  Fold Best -> T1: {best_acc_t1:.2f} | T2: {best_acc_t2:.2f}")
        fold_results_t1.append(best_acc_t1)
        fold_acc_t2.append(best_acc_t2)
        
        # Save One Model (Optional, saves last fold model)
        torch.save(model.state_dict(), f'task1_2_seq/model_fold{fold}.pth')

    print("\n=== FINAL RESULTS (5-Fold CV) ===")
    print(f"Task 1 (Binary) Avg Acc: {np.mean(fold_results_t1)*100:.2f}%")
    print(f"Task 2 (Multi)  Avg Acc: {np.mean(fold_acc_t2)*100:.2f}%")
    
    # Save a generic model for inference
    torch.save(model.state_dict(), args.model_out)
    print(f"Final model saved to {args.model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--model_out', type=str, default='task1_2_seq/model_seq.pth')
    args = parser.parse_args()
    train(args)
