"""
=========================================================
 Student Engagement Recognition - Training Script
 Task 1: Visual Binary Classification (NUM_CLASSES=2)
 Task 2: Visual Multi-Class Classification (NUM_CLASSES=4)
 
 Self-contained training script for judges.
 
 Usage:
   # Task 1 (Binary): 0,0.33 -> Low | 0.66,1 -> High
   python task1_2_visual/train.py --task 1 --data_dir dataset/train --labels dataset/train/labels_train.xlsx

   # Task 2 (Multi-class): 0->Distracted, 0.33->Disengaged, 0.66->NomEngaged, 1->HighEngaged
   python task1_2_visual/train.py --task 2 --data_dir dataset/train --labels dataset/train/labels_train.xlsx

 Output:
   Saves best model to task1_2_visual/model_task1.pth or model_task2.pth
=========================================================
"""

import os
import sys
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sk_f1
from tqdm import tqdm
from collections import Counter, defaultdict
import cv2

# ========================= CONFIG =========================
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased batch size for more stable gradients
EPOCHS = 60      # More epochs since we are only training the head
LEARNING_RATE = 1e-3 # Higher LR because we are training from scratch (Head)
FRAMES_PER_VIDEO_TRAIN = 100
FRAMES_PER_VIDEO_VAL = 500
SEED = 42

# ========================= MODEL =========================
class VisualEngagementModel(nn.Module):
    """ResNet18-based classifier for engagement recognition."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        
        # --------------------------------------------------------
        # OPTIMIZATION: STRONG FREEZING
        # With only ~60 videos, fine-tuning conv layers leads to massive overfitting.
        # We freeze the ENTIRE backbone and only train the classification head.
        # --------------------------------------------------------
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Lightweight Head with stronger Regularization
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),        # stabilize learning
            nn.ReLU(),
            nn.Dropout(0.6),            # high dropout
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_param_groups(self):
        # All trainable parameters are in the head (fc)
        return [], [p for p in self.parameters() if p.requires_grad]

# ========================= DATASET =========================
class EngagementDataset(Dataset):
    """Frame-level dataset with video-level tracking."""
    def __init__(self, video_paths, labels, num_classes, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.num_classes = num_classes
        self.video_data = []
        
        for video_path, label in zip(video_paths, labels):
            if num_classes == 2:
                target = 0 if label <= 0.33 else 1
            else:
                label_rounded = round(float(label), 2)
                label_map = {0.0: 0, 0.33: 1, 0.66: 2, 1.0: 3}
                target = label_map.get(label_rounded, int(round(float(label) * 3)))
            
            frames = []
            if os.path.exists(video_path):
                frames = sorted([os.path.join(video_path, f)
                                for f in os.listdir(video_path)
                                if f.endswith(('.jpg', '.png'))])
            if not frames:
                continue
            self.video_data.append((frames, target))
        
        self._build_samples()
        n = FRAMES_PER_VIDEO_TRAIN if phase == 'train' else FRAMES_PER_VIDEO_VAL
        print(f"  [{phase.upper()}] {len(self.samples)} samples from {len(self.video_data)} videos (~{n} frames/video)")
    
    def get_video_labels(self):
        return {i: t for i, (_, t) in enumerate(self.video_data)}
    
    def _build_samples(self):
        self.samples = []
        n = FRAMES_PER_VIDEO_TRAIN if self.phase == 'train' else FRAMES_PER_VIDEO_VAL
        for vid_idx, (all_frames, target) in enumerate(self.video_data):
            if len(all_frames) <= n:
                selected = all_frames
            elif self.phase == 'train':
                selected = random.sample(all_frames, n)
            else:
                indices = np.linspace(0, len(all_frames) - 1, n, dtype=int)
                selected = [all_frames[i] for i in indices]
            for fp in selected:
                self.samples.append((fp, target, vid_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def reshuffle(self):
        if self.phase == 'train':
            self._build_samples()
    
    def __getitem__(self, idx):
        img_path, target, vid_idx = self.samples[idx]
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long), vid_idx

# ========================= MIXUP =========================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========================= TRAIN / VALIDATE =========================
def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    mix_alpha = 0.1 if num_classes == 4 else 0.2
    
    for images, labels, _ in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        mixed, ya, yb, lam = mixup_data(images, labels, alpha=mix_alpha)
        optimizer.zero_grad()
        outputs = model(mixed)
        loss = mixup_criterion(criterion, outputs, ya, yb, lam)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += lam * pred.eq(ya).sum().item() + (1 - lam) * pred.eq(yb).sum().item()
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, val_dataset, num_classes,
             val_paths=None, verbose=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    video_probs = defaultdict(list)
    video_labels = val_dataset.get_video_labels()
    
    if num_classes == 4:
        label_names = {0: 'Distracted', 1: 'Disengaged', 2: 'NomEngaged', 3: 'HighEngaged'}
    else:
        label_names = {0: 'Distracted', 1: 'Engaged'}
    
    with torch.no_grad():
        for images, labels, vid_indices in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            # TTA: horizontal flip
            probs_flip = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = (probs + probs_flip) / 2.0
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for prob, vi in zip(probs.cpu(), vid_indices.tolist()):
                video_probs[vi].append(prob)
    
    frame_acc = 100. * correct / total
    video_acc = 0.0
    
    if video_labels:
        video_correct = 0
        video_details = []
        for vi, plist in video_probs.items():
            avg = torch.stack(plist).mean(dim=0)
            pc = avg.argmax().item()
            tc = video_labels[vi]
            ok = pc == tc
            if ok:
                video_correct += 1
            vname = os.path.basename(val_paths[vi]) if val_paths else f"video_{vi}"
            conf = avg[pc].item() * 100
            d = {'name': vname, 'true': tc, 'pred': pc, 'confidence': conf, 'correct': ok}
            for ci in range(num_classes):
                d[f'prob_c{ci}'] = avg[ci].item() * 100
            video_details.append(d)
        video_acc = 100. * video_correct / len(video_probs)
        
        if verbose:
            # F1-Macro and per-class accuracy
            true_list = [video_labels[vi] for vi in video_probs]
            pred_list = [torch.stack(video_probs[vi]).mean(0).argmax().item() for vi in video_probs]
            f1m = sk_f1(true_list, pred_list, average='macro', zero_division=0)
            print(f"\n  F1-Macro: {f1m:.4f}")
            if num_classes == 4:
                ct, cc = {}, {}
                for t, p in zip(true_list, pred_list):
                    ct[t] = ct.get(t, 0) + 1
                    if t == p: cc[t] = cc.get(t, 0) + 1
                for ci in range(num_classes):
                    tot = ct.get(ci, 0)
                    cor = cc.get(ci, 0)
                    print(f"  Class {ci} ({label_names[ci]:11s}): {cor}/{tot} = {cor/tot*100 if tot else 0:.1f}%")
            
            print("\n" + "=" * 80)
            print("PER-VIDEO DIAGNOSIS (sorted by confidence)")
            print("=" * 80)
            video_details.sort(key=lambda x: x['confidence'])
            for v in video_details:
                st = 'OK' if v['correct'] else 'WRONG'
                bl = 'BORDERLINE' if v['confidence'] < 40 else ''
                ps = ' | '.join([f"C{i}:{v.get(f'prob_c{i}',0):4.1f}%" for i in range(num_classes)])
                print(f"  [{st:5s}] {v['name']:20s} | True: {label_names[v['true']]:11s} | "
                      f"Pred: {label_names[v['pred']]:11s} | Conf: {v['confidence']:5.1f}% | {ps} {bl}")
            print("=" * 80 + "\n")
    
    return running_loss / len(loader), frame_acc, video_acc

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Train Visual Engagement Model (Task 1 & 2)")
    parser.add_argument("--task", type=int, default=2, choices=[1, 2],
                        help="Task number: 1=Binary, 2=Multi-class (default: 2)")
    parser.add_argument("--data_dir", type=str, default="dataset/train",
                        help="Path to processed frames directory")
    parser.add_argument("--labels", type=str, default="dataset/train/labels_train.xlsx",
                        help="Path to labels Excel file")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None,
                        help="Output model path (default: task1_2_visual/model.pth)")
    args = parser.parse_args()
    
    num_classes = 2 if args.task == 1 else 4
    
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task {args.task} | Classes: {num_classes} | Seed: {args.seed} | Device: {device}")
    
    # Transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # Added crop for robustness
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25))
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load labels
    df = pd.read_excel(args.labels)
    video_names = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].tolist()
    
    # Map to processed frame folders
    processed_paths, valid_labels = [], []
    for name, lbl in zip(video_names, labels):
        folder = os.path.join(args.data_dir, os.path.splitext(name)[0])
        if os.path.exists(folder) and len(os.listdir(folder)) > 0:
            processed_paths.append(folder)
            valid_labels.append(lbl)
    print(f"Found {len(processed_paths)} valid videos out of {len(df)}")
    
    # Class distribution
    if num_classes == 4:
        dist = Counter([round(float(l), 2) for l in valid_labels])
        print(f"  Distribution: {dict(sorted(dist.items()))}")
    
    # Train/Val split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            processed_paths, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            processed_paths, valid_labels, test_size=0.2, random_state=42)
    
    # Datasets
    train_ds = EngagementDataset(X_train, y_train, num_classes, train_transform, 'train')
    val_ds = EngagementDataset(X_val, y_val, num_classes, val_transform, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = VisualEngagementModel(num_classes=num_classes).to(device)
    
    # Class weights (tuned for small dataset imbalance)
    if num_classes == 2:
        weights = torch.tensor([1.2, 1.0]).to(device)
    else:
        # Heavily penalize ignoring the rare classes (0 and 2 often confused)
        weights = torch.tensor([4.0, 2.0, 1.5, 1.0]).to(device)
    
    smoothing = 0.2 if num_classes == 4 else 0.15 # Increased smoothing
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    
    # Optimizer - simplified since backbone is frozen
    _, hp = model.get_param_groups()
    optimizer = optim.Adam(hp, lr=LEARNING_RATE, weight_decay=1e-2) # Stronger weight decay (0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Output path (task-specific so Task 1 and Task 2 don't overwrite each other)
    if args.output:
        save_path = args.output
    else:
        model_name = f'model_task{args.task}.pth'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    patience, no_improve = 15, 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_ds.reshuffle()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, frame_acc, video_acc = validate(model, val_loader, criterion, device, val_ds, num_classes, val_paths=X_val)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Frame Acc: {frame_acc:.2f}% | Video Acc: {video_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
        
        if video_acc > best_acc:
            best_acc = video_acc
            no_improve = 0
            try:
                import io, shutil
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                tmp = '/tmp/model_visual.pth'
                with open(tmp, 'wb') as f:
                    f.write(buf.getvalue())
                try:
                    shutil.copy2(tmp, save_path)
                    print(f"Saved Best Model to {save_path}")
                except OSError:
                    print(f"WARNING: Disk quota! Model at {tmp}")
            except Exception as e:
                print(f"WARNING: Save failed: {e}")
            
            validate(model, val_loader, criterion, device, val_ds, num_classes, val_paths=X_val, verbose=True)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                print(f"\nEarly Stopping! Best: {best_acc:.2f}%")
                break
    
    qual = 70.0 if num_classes == 2 else 65.0
    status = "QUALIFIED" if best_acc >= qual else "NOT QUALIFIED"
    print(f"\nTraining Complete! Best Val Acc: {best_acc:.2f}% [{status}] (threshold: {qual}%)")

if __name__ == "__main__":
    main()
