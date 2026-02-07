"""
=========================================================
 Student Engagement Recognition - LSTM Inference Script
 Task 1: Visual Binary Classification  (Accuracy + F1-Score)
 Task 2: Visual Multi-Class Classification (Accuracy Class-wise + F1-Macro + Confusion Matrix)

 Self-contained inference script using LSTM + MediaPipe features.

 Usage:
   # Task 1 (Binary) - with labels for evaluation
   python task1_2_seq/inference.py --task 1 --test_dir dataset/test --labels dataset/test/labels_test.xlsx --model task1_2_seq/model_seq.pth

   # Task 2 (Multi-class) - with labels for evaluation
   python task1_2_seq/inference.py --task 2 --test_dir dataset/test --labels dataset/test/labels_test.xlsx --model task1_2_seq/model_seq.pth

   # Without labels (predictions only)
   python task1_2_seq/inference.py --task 1 --test_dir dataset/test --model task1_2_seq/model_seq.pth

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

import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import mediapipe as mp
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

# ========================= CONFIG =========================
FPS_TARGET = 6
SEQ_LEN = 100
MAX_FRAMES = 9000  # Cap at 5 min @30fps to prevent RAM overflow
EXTENSIONS = ['.mp4', '.MP4', '.avi', '.AVI', '.webm', '.wmv', '.mov', '.MOV']

# ========================= FEATURE EXTRACTION =========================

def get_head_pose(landmarks, shape):
    img_h, img_w, _ = shape
    face_3d = []
    face_2d = []
    key_landmarks = [1, 199, 33, 263, 61, 291]

    for idx, lm in enumerate(landmarks):
        if idx in key_landmarks:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return 0, 0, 0

    rmat, jac = cv2.Rodrigues(rot_vec)
    result = cv2.RQDecomp3x3(rmat)
    angles = result[0]
    return angles[0] * 360, angles[1] * 360, angles[2] * 360


def get_eye_ratio(landmarks, eye_indices):
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[3]]
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[2]]

    v_dist = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    h_dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
    if h_dist == 0:
        return 0
    return v_dist / h_dist


# Global FaceMesh — reuse across all videos to prevent TFLite memory leak
_face_mesh_instance = None

def _get_face_mesh():
    global _face_mesh_instance
    if _face_mesh_instance is None:
        _face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    return _face_mesh_instance


def extract_features(video_path, verbose=False):
    """Extract 9-dim features from video using MediaPipe Face Mesh."""
    face_mesh = _get_face_mesh()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(fps / FPS_TARGET)))

    if verbose:
        print(f"    FPS: {fps:.1f}, Frames: {total_frames}, Interval: {frame_interval}")

    features = []
    frame_count = 0
    faces_found = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if frame_count > MAX_FRAMES:
            break  # Cap to prevent RAM overflow on long videos

        if frame_count % frame_interval == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            results = face_mesh.process(image_rgb)
            del image_rgb  # Free immediately

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pitch, yaw, roll = get_head_pose(lm, (img_h, img_w, 3))

                le_x = (lm[33].x + lm[133].x) / 2
                le_y = (lm[33].y + lm[133].y) / 2
                re_x = (lm[362].x + lm[263].x) / 2
                re_y = (lm[362].y + lm[263].y) / 2

                left_ear = get_eye_ratio(lm, [33, 159, 133, 145])
                right_ear = get_eye_ratio(lm, [362, 386, 263, 374])

                feat = [pitch, yaw, roll, le_x, le_y, re_x, re_y, left_ear, right_ear]
                features.append(feat)
                faces_found += 1
            else:
                features.append([0] * 9)

        frame_count += 1

    cap.release()
    # Don't close face_mesh — it's reused across videos

    if len(features) == 0:
        return None

    features = np.array(features)
    if verbose:
        print(f"    Features: {len(features)} frames ({faces_found} with face)")
    return features


# ========================= MODEL =========================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1), weights


class EngagementLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.3))
        self.head_binary = nn.Linear(32, 1)
        self.head_regression = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        feat = self.fc(context)
        return self.head_binary(feat), self.head_regression(feat)


# ========================= PREDICTION =========================

LABEL_NAMES_BIN = {0: 'Distracted', 1: 'Engaged'}
LABEL_NAMES_MULTI = {0: 'Distracted', 1: 'Disengaged', 2: 'NomEngaged', 3: 'HighEngaged'}


def bin_prediction(val):
    """Map regression value to 4-class label."""
    if val < 0.165:
        return 0
    elif val < 0.5:
        return 1
    elif val < 0.835:
        return 2
    else:
        return 3


def predict(features, model, device):
    """Run model prediction on extracted features."""
    # Normalize
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)

    # Pad/truncate to SEQ_LEN
    L, D = features.shape
    if L < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - L, D))
        features = np.vstack([features, pad])
    elif L > SEQ_LEN:
        indices = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
        features = features[indices]

    feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out_bin, out_reg = model(feat_tensor)

        # Task 1: Binary
        prob = torch.sigmoid(out_bin).item()
        binary_pred = 1 if prob > 0.5 else 0

        # Task 2: Multi-class (regression -> binning)
        reg_val = out_reg.item()
        reg_val_clamped = max(0.0, min(1.0, reg_val))
        multi_class = bin_prediction(reg_val_clamped)

    return binary_pred, prob, multi_class, reg_val_clamped


def find_video(test_dir, name):
    """Find video file with any supported extension."""
    exact = os.path.join(test_dir, name)
    if os.path.isfile(exact):
        return exact
    base = os.path.splitext(name)[0]
    for ext in EXTENSIONS:
        p = os.path.join(test_dir, base + ext)
        if os.path.isfile(p):
            return p
    return None


# ========================= MAIN =========================

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Inference for Student Engagement Recognition")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2],
                        help="Task: 1=Binary, 2=Multi-class")
    parser.add_argument("--test_dir", type=str, default="dataset/test",
                        help="Path to test videos folder (default: dataset/test)")
    parser.add_argument("--labels", type=str, default="dataset/test/labels_test.xlsx",
                        help="Path to labels Excel/CSV (col0=name, col1=label). "
                             "Use --no_eval to skip evaluation.")
    parser.add_argument("--model", type=str, default="task1_2_seq/model_seq.pth",
                        help="Path to trained LSTM model weights")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: results_task{N}_seq.csv)")
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip evaluation (prediction-only, ignore labels)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-video feature extraction details")
    args = parser.parse_args()

    num_classes = 2 if args.task == 1 else 4
    label_names = LABEL_NAMES_BIN if args.task == 1 else LABEL_NAMES_MULTI
    output_csv = args.output or f"results_task{args.task}_seq.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  Student Engagement Recognition - LSTM Inference")
    print("=" * 80)
    print(f"  Task:       {args.task} ({'Binary' if args.task == 1 else 'Multi-class 4-way'})")
    print(f"  Model:      LSTM + Attention (MediaPipe 9-dim features)")
    print(f"  Weights:    {args.model}")
    print(f"  Test Dir:   {args.test_dir}")
    print(f"  Labels:     {args.labels if not args.no_eval else 'Disabled (--no_eval)'}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_csv}")
    print("=" * 80)

    # Validate paths
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.isdir(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Load model
    model = EngagementLSTM().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("Model loaded successfully!\n")

    # ---- MODE 1: With labels (evaluation) ----
    has_labels = not args.no_eval and os.path.exists(args.labels)
    if has_labels:
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
            true_labels = [lmap.get(round(float(l), 2), int(round(float(l) * 3)))
                           for l in raw_labels]

        print(f"Test videos in labels: {len(video_names)}")
        print("-" * 80)

    # ---- MODE 2: Without labels (scan directory) ----
    if not has_labels:
        video_names = []
        for f in sorted(os.listdir(args.test_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in [e.lower() for e in EXTENSIONS]:
                video_names.append(f)
        true_labels = None
        raw_labels = [None] * len(video_names)
        print(f"Found {len(video_names)} videos in {args.test_dir}")
        print("-" * 80)

    if len(video_names) == 0:
        print("ERROR: No videos found!")
        sys.exit(1)

    # ---- Process videos ----
    results_data = []
    skipped = 0

    for i, name in enumerate(video_names):
        video_path = find_video(args.test_dir, name)

        if video_path is None:
            print(f"  [{i+1}/{len(video_names)}] SKIP - Not found: {name}")
            skipped += 1
            continue

        # Extract features
        features = extract_features(video_path, verbose=args.verbose)
        if features is None or len(features) == 0:
            print(f"  [{i+1}/{len(video_names)}] FAIL - No features: {name}")
            skipped += 1
            gc.collect()
            continue

        # Predict
        binary_pred, binary_prob, multi_class, reg_val = predict(features, model, device)
        del features  # Free RAM immediately after prediction
        gc.collect()

        # Select prediction based on task
        if args.task == 1:
            pred = binary_pred
            conf = binary_prob if binary_pred == 1 else (1 - binary_prob)
        else:
            pred = multi_class
            conf = 1.0 - abs(reg_val - [0.0, 0.33, 0.66, 1.0][multi_class])

        # Print result
        if has_labels:
            true_lbl = true_labels[i]
            raw_lbl = raw_labels[i]
            st = "OK" if pred == true_lbl else "XX"
            print(f"  [{i+1}/{len(video_names)}] {st:2s} {name:30s} | "
                  f"True: {label_names[true_lbl]:11s} (raw={raw_lbl}) | "
                  f"Pred: {label_names[pred]:11s} ({conf*100:.1f}%)")
        else:
            print(f"  [{i+1}/{len(video_names)}] {name:30s} | "
                  f"Pred: {label_names[pred]:11s} ({conf*100:.1f}%)")

        row = {
            'video': name,
            'predicted_class': pred,
            'predicted_name': label_names[pred],
            'binary_pred': binary_pred,
            'binary_prob': round(binary_prob, 4),
            'multi_class': multi_class,
            'regression_val': round(reg_val, 4),
        }
        if has_labels:
            row['true_raw_label'] = raw_labels[i]
            row['true_class'] = true_labels[i]
            row['true_name'] = label_names[true_labels[i]]
            row['correct'] = pred == true_labels[i]

        results_data.append(row)

    # ---- METRICS ----
    print("\n" + "=" * 80)

    if not results_data:
        print("No predictions made. Check paths and videos.")
        return

    # Save CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False)
    print(f"  Results saved to: {output_csv}")
    print(f"  Processed: {len(results_data)} | Skipped: {skipped}")

    if not has_labels:
        print("\n  No labels provided - evaluation skipped.")
        print("  Run with --labels to get accuracy/F1 metrics.")
        print("=" * 80)
        return

    true_eval = [r['true_class'] for r in results_data]
    pred_eval = [r['predicted_class'] for r in results_data]
    acc = accuracy_score(true_eval, pred_eval)

    if args.task == 1:
        # ----- TASK 1 METRICS -----
        f1_bin = f1_score(true_eval, pred_eval, average='binary')
        f1_mac = f1_score(true_eval, pred_eval, average='macro')
        cm = confusion_matrix(true_eval, pred_eval)

        print("               EVALUATION RESULTS - TASK 1")
        print("=" * 80)
        print(f"  Task:           Visual Binary Classification")
        print(f"  Model:          LSTM + Attention (MediaPipe 9-dim features)")
        print(f"  Videos Tested:  {len(results_data)} | Skipped: {skipped}")
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
        # ----- TASK 2 METRICS -----
        f1_mac = f1_score(true_eval, pred_eval, average='macro', zero_division=0)
        f1_w = f1_score(true_eval, pred_eval, average='weighted', zero_division=0)
        cm = confusion_matrix(true_eval, pred_eval, labels=[0, 1, 2, 3])

        # Class-wise accuracy
        cls_c, cls_t = {}, {}
        for t, p in zip(true_eval, pred_eval):
            cls_t[t] = cls_t.get(t, 0) + 1
            if t == p:
                cls_c[t] = cls_c.get(t, 0) + 1

        print("               EVALUATION RESULTS - TASK 2")
        print("=" * 80)
        print(f"  Task:           Visual Multi-Class Classification")
        print(f"  Model:          LSTM + Attention (MediaPipe 9-dim features)")
        print(f"  Classes:        4 (Distracted, Disengaged, NomEngaged, HighEngaged)")
        print(f"  Videos Tested:  {len(results_data)} | Skipped: {skipped}")
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
        hdr = "              " + "  ".join(
            [f"Pred:{label_names[i][:5]:>6s}" for i in range(4)])
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
    print("=" * 80)


if __name__ == "__main__":
    main()
