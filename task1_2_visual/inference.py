"""
=========================================================
 Student Engagement Recognition - Inference Script
 Task 1: Visual Binary Classification  (Accuracy + F1-Score)
 Task 2: Visual Multi-Class Classification (Accuracy Class-wise + F1-Macro + Confusion Matrix)
 
 Self-contained inference script for judges.
 
 Usage:
   # Task 1 (Binary)
   python task1_2_visual/inference.py --task 1 --test_dir dataset/test --labels dataset/test/labels_test.xlsx --model task1_2_visual/model.pth

   # Task 2 (Multi-class)
   python task1_2_visual/inference.py --task 2 --test_dir dataset/test --labels dataset/test/labels_test.xlsx --model task1_2_visual/model.pth

 Output:
   - Per-video predictions (printed + saved to CSV)
   - Task 1: Accuracy, F1-Score, Confusion Matrix
   - Task 2: Accuracy (Class-wise), F1-Macro, Confusion Matrix
=========================================================
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
import mediapipe as mp

# ========================= CONFIG =========================
IMG_SIZE = 224
NUM_FRAMES_DEFAULT = 30

# ========================= MODEL (same as train.py) =========================
class VisualEngagementModel(nn.Module):
    """ResNet18-based classifier for engagement recognition."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # No pretrained needed for inference
        in_features = self.backbone.fc.in_features
        
        for name, param in self.backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        hidden_dim = 256 if num_classes > 2 else 128
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ========================= FACE DETECTOR =========================
class FaceDetector:
    """MediaPipe face detector matching training pipeline."""
    def __init__(self, confidence=0.5, padding=0):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=confidence)
        self.padding = padding
    
    def extract_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = max(0, int(bbox.xmin * iw) - self.padding)
            y = max(0, int(bbox.ymin * ih) - self.padding)
            w = min(iw - x, int(bbox.width * iw) + 2 * self.padding)
            h = min(ih - y, int(bbox.height * ih) + 2 * self.padding)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                return face
        return None

# ========================= PREDICTION =========================
def predict_video(video_path, model, face_detector, transform, device,
                  num_classes, num_frames=30, tta=True):
    """
    Predict engagement for a single video.
    Uses probability averaging + Test-Time Augmentation (horizontal flip).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return -1, 0.0, []
    
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    all_probs = []
    per_frame_preds = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = face_detector.extract_face(frame)
        if face is None:
            continue
        
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(face_rgb)
        inp = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = F.softmax(model(inp), dim=1)
            if tta:
                probs_flip = F.softmax(model(torch.flip(inp, dims=[3])), dim=1)
                probs = (probs + probs_flip) / 2.0
            all_probs.append(probs.cpu())
            per_frame_preds.append(probs.argmax(dim=1).item())
    
    cap.release()
    if not all_probs:
        return -1, 0.0, []
    
    avg_prob = torch.cat(all_probs, dim=0).mean(dim=0)
    pred_class = avg_prob.argmax().item()
    confidence = avg_prob[pred_class].item()
    class_probs = [avg_prob[i].item() for i in range(num_classes)]
    
    return pred_class, confidence, class_probs

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Inference for Visual Engagement Model")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2],
                        help="Task: 1=Binary, 2=Multi-class")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test videos folder")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels Excel/CSV (col0=name, col1=label)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (default: task1_2_visual/model_task{N}.pth)")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: results_task{N}.csv)")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()
    
    num_classes = 2 if args.task == 1 else 4
    output_csv = args.output or f"results_task{args.task}.csv"
    
    # Default model path is task-specific
    if args.model is None:
        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"model_task{args.task}.pth")
    
    if num_classes == 2:
        label_names = {0: 'Distracted', 1: 'Engaged'}
    else:
        label_names = {0: 'Distracted', 1: 'Disengaged', 2: 'NomEngaged', 3: 'HighEngaged'}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task {args.task} | Classes: {num_classes} | Device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = VisualEngagementModel(num_classes=num_classes)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    face_detector = FaceDetector(confidence=0.5, padding=0)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load labels
    if args.labels.endswith('.csv'):
        df = pd.read_csv(args.labels)
    else:
        df = pd.read_excel(args.labels)
    
    video_names = df.iloc[:, 0].astype(str).tolist()
    raw_labels = df.iloc[:, 1].tolist()
    
    # Map labels
    if num_classes == 2:
        true_labels = [0 if lbl <= 0.33 else 1 for lbl in raw_labels]
    else:
        lmap = {0.0: 0, 0.33: 1, 0.66: 2, 1.0: 3}
        true_labels = [lmap.get(round(float(l), 2), int(round(float(l)*3))) for l in raw_labels]
    
    print(f"Test videos: {len(video_names)}")
    print(f"Test directory: {args.test_dir}")
    print(f"TTA: {'OFF' if args.no_tta else 'ON'}")
    print("=" * 80)
    
    EXTENSIONS = ['.mp4', '.MP4', '.avi', '.AVI', '.webm', '.wmv', '.mov']
    results_data = []
    skipped = 0
    
    for i, (name, true_lbl, raw_lbl) in enumerate(zip(video_names, true_labels, raw_labels)):
        video_path = None
        base = os.path.splitext(name)[0]
        exact = os.path.join(args.test_dir, name)
        if os.path.isfile(exact):
            video_path = exact
        else:
            for ext in EXTENSIONS:
                p = os.path.join(args.test_dir, base + ext)
                if os.path.isfile(p):
                    video_path = p
                    break
        
        if video_path is None:
            print(f"  [{i+1}/{len(video_names)}] SKIP - Not found: {name}")
            skipped += 1
            continue
        
        pred, conf, cprobs = predict_video(
            video_path, model, face_detector, transform, device,
            num_classes, args.num_frames, tta=not args.no_tta)
        
        if pred == -1:
            print(f"  [{i+1}/{len(video_names)}] FAIL - No faces: {name}")
            skipped += 1
            continue
        
        st = "OK" if pred == true_lbl else "XX"
        if num_classes == 4:
            ps = ' '.join([f"C{j}:{cprobs[j]*100:4.1f}%" for j in range(4)])
        else:
            ps = f"P(Engaged):{cprobs[1]*100:.1f}%"
        print(f"  [{i+1}/{len(video_names)}] {st:2s} {name:30s} | "
              f"True: {label_names[true_lbl]:11s} (raw={raw_lbl}) | "
              f"Pred: {label_names[pred]:11s} ({conf*100:.1f}%) | {ps}")
        
        row = {
            'video': name, 'true_raw_label': raw_lbl,
            'true_class': true_lbl, 'predicted_class': pred,
            'confidence': round(conf, 4),
            'true_name': label_names[true_lbl],
            'predicted_name': label_names[pred],
            'correct': pred == true_lbl
        }
        if num_classes == 4:
            for ci in range(4):
                row[f'prob_class_{ci}'] = round(cprobs[ci], 4)
        results_data.append(row)
    
    # ==================== METRICS ====================
    if not results_data:
        print("\nNo predictions made. Check paths.")
        return
    
    true_eval = [r['true_class'] for r in results_data]
    pred_eval = [r['predicted_class'] for r in results_data]
    
    acc = accuracy_score(true_eval, pred_eval)
    
    print("\n" + "=" * 80)
    if args.task == 1:
        # ----- TASK 1 METRICS (PS: Accuracy + F1-Score) -----
        f1_bin = f1_score(true_eval, pred_eval, average='binary')
        f1_mac = f1_score(true_eval, pred_eval, average='macro')
        cm = confusion_matrix(true_eval, pred_eval)
        
        print("               EVALUATION RESULTS - TASK 1")
        print("=" * 80)
        print(f"  Task:           Visual Binary Classification")
        print(f"  Model:          ResNet18 (Transfer Learning)")
        print(f"  Videos Tested:  {len(results_data)} | Skipped: {skipped}")
        print(f"  Frames/Video:   {args.num_frames} | TTA: {'OFF' if args.no_tta else 'ON'}")
        print("-" * 80)
        print(f"  ACCURACY:       {acc*100:.2f}%")
        print(f"  F1-SCORE:       {f1_bin:.4f}")
        print(f"  F1 (Macro):     {f1_mac:.4f}")
        print("-" * 80)
        print(f"  Confusion Matrix:")
        print(f"                    Pred: Distracted  Pred: Engaged")
        print(f"  True: Distracted      {cm[0][0]:^12d}    {cm[0][1]:^12d}")
        print(f"  True: Engaged         {cm[1][0]:^12d}    {cm[1][1]:^12d}")
        print("-" * 80)
        print("\n  Classification Report:")
        print(classification_report(true_eval, pred_eval,
                                     target_names=["Distracted", "Engaged"]))
        qual = 70.0
        
    else:
        # ----- TASK 2 METRICS (PS: Accuracy Class-wise + F1-Macro + Confusion Matrix) -----
        f1_mac = f1_score(true_eval, pred_eval, average='macro', zero_division=0)
        f1_w = f1_score(true_eval, pred_eval, average='weighted', zero_division=0)
        cm = confusion_matrix(true_eval, pred_eval, labels=[0, 1, 2, 3])
        
        # Class-wise accuracy
        cls_c, cls_t = {}, {}
        for t, p in zip(true_eval, pred_eval):
            cls_t[t] = cls_t.get(t, 0) + 1
            if t == p: cls_c[t] = cls_c.get(t, 0) + 1
        
        print("               EVALUATION RESULTS - TASK 2")
        print("=" * 80)
        print(f"  Task:           Visual Multi-Class Classification")
        print(f"  Model:          ResNet18 (Transfer Learning)")
        print(f"  Classes:        4 (Distracted, Disengaged, NomEngaged, HighEngaged)")
        print(f"  Videos Tested:  {len(results_data)} | Skipped: {skipped}")
        print(f"  Frames/Video:   {args.num_frames} | TTA: {'OFF' if args.no_tta else 'ON'}")
        print("-" * 80)
        print(f"  OVERALL ACCURACY:   {acc*100:.2f}%")
        print(f"  F1-SCORE (Macro):   {f1_mac:.4f}")
        print(f"  F1-SCORE (Weight):  {f1_w:.4f}")
        print("-" * 80)
        print(f"  CLASS-WISE ACCURACY:")
        for ci in range(4):
            tot = cls_t.get(ci, 0)
            cor = cls_c.get(ci, 0)
            ca = (cor / tot * 100) if tot > 0 else 0
            print(f"    Class {ci} ({label_names[ci]:11s}): {cor}/{tot} = {ca:.1f}%")
        print("-" * 80)
        print(f"  Confusion Matrix:")
        hdr = "              " + "  ".join([f"Pred:{label_names[i][:5]:>6s}" for i in range(4)])
        print(hdr)
        for i in range(4):
            row = f"  True:{label_names[i][:5]:>6s}"
            for j in range(4):
                row += f"  {cm[i][j]:>6d}"
            print(row)
        print("-" * 80)
        print("\n  Classification Report:")
        tnames = [label_names[i] for i in range(4)]
        print(classification_report(true_eval, pred_eval,
                                     target_names=tnames, labels=[0, 1, 2, 3],
                                     zero_division=0))
        qual = 65.0
    
    print("=" * 80)
    if acc * 100 >= qual:
        print(f"  >> QUALIFIED! Accuracy {acc*100:.2f}% >= {qual}%")
    else:
        print(f"  >> NOT QUALIFIED. Accuracy {acc*100:.2f}% < {qual}%")
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False)
    print(f"\n  Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
