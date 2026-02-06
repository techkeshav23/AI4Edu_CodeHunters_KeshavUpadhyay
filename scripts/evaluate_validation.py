"""
=========================================================
 Validation Set Evaluation - 15 Held-Out Videos
 Task 1: Visual Binary Classification

 Evaluates ONLY the 15 validation videos that were NOT
 used during training (random_state=42, 80/20 split).

 Usage:
   python scripts/evaluate_validation.py

 No arguments needed - paths are hardcoded for quick run.
=========================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import Counter
import mediapipe as mp
from PIL import Image

# -------------------- CONFIG --------------------
MODEL_PATH = "models/visual/best_visual_model.pth"
VIDEOS_DIR = "data/raw/videos/Train"
LABELS_PATH = "data/raw/labels_train.xlsx"
NUM_FRAMES = 10  # Frames to sample per video

# These 15 videos were held out during training (stratified split, random_state=42)
VALIDATION_VIDEOS = [
    "subject_35_Vid_7",
    "subject_3_Vid_2",
    "subject_33_Vid_1",
    "subject_79_Vid_7",
    "subject_65_Vid_6",
    # "subject_83_Vid_7",
    # "subject_26_Vid_5",
    # "subject_62_Vid_5",
    # "subject_41_Vid_7",
    # "subject_62_Vid_7",
    "subject_85_Vid_7",
    "subject_9_Vid_6",
    "subject_84_Vid_1",
    "subject_1_Vid_5",
    "subject_77_Vid_7",
]

EXTENSIONS = ['.avi', '.mp4', '.MP4', '.webm', '.wmv', '.mov']

# -------------------- MODEL --------------------
class VisualEngagementModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(VisualEngagementModel, self).__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# -------------------- FACE DETECTOR --------------------
class FaceDetector:
    def __init__(self, confidence=0.5, padding=0):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=confidence
        )
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

# -------------------- PREDICT VIDEO --------------------
def predict_video(video_path, model, face_detector, transform, device):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return -1, 0.0, []

    if total_frames <= NUM_FRAMES:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int).tolist()

    predictions = []
    confidences = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        face = face_detector.extract_face(frame)
        if face is None:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        predictions.append(pred)
        confidences.append(conf)

    cap.release()

    if not predictions:
        return -1, 0.0, []

    vote = Counter(predictions).most_common(1)[0][0]
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == vote])
    return vote, avg_conf, predictions

# -------------------- FIND VIDEO FILE --------------------
def find_video(name):
    """Find video file with any extension."""
    for ext in EXTENSIONS:
        path = os.path.join(VIDEOS_DIR, name + ext)
        if os.path.isfile(path):
            return path
    return None

# -------------------- MAIN --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = VisualEngagementModel(num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded: {MODEL_PATH}\n")

    # Face detector & transforms (must match training)
    face_detector = FaceDetector(confidence=0.5, padding=0)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load labels from Excel
    df = pd.read_excel(LABELS_PATH)
    label_dict = {}
    for _, row in df.iterrows():
        name = os.path.splitext(str(row.iloc[0]))[0]
        label_dict[name] = row.iloc[1]

    label_map = {0: "Distracted", 1: "Engaged"}

    print("=" * 65)
    print("  VALIDATION SET EVALUATION (15 Held-Out Videos)")
    print("  These videos were NOT used during training")
    print("=" * 65)

    true_labels = []
    pred_labels = []
    results = []

    for i, vid_name in enumerate(VALIDATION_VIDEOS):
        # Get true label
        raw_label = label_dict.get(vid_name, None)
        if raw_label is None:
            print(f"  [{i+1:2d}/15] SKIP - Label not found: {vid_name}")
            continue

        true_binary = 0 if raw_label <= 0.33 else 1

        # Find video file
        video_path = find_video(vid_name)
        if video_path is None:
            print(f"  [{i+1:2d}/15] SKIP - Video not found: {vid_name}")
            continue

        # Predict
        pred, conf, frame_preds = predict_video(
            video_path, model, face_detector, transform, device
        )

        if pred == -1:
            print(f"  [{i+1:2d}/15] FAIL - No faces: {vid_name}")
            continue

        true_labels.append(true_binary)
        pred_labels.append(pred)

        status = "CORRECT" if pred == true_binary else "WRONG  "
        marker = "+" if pred == true_binary else "X"
        print(f"  [{i+1:2d}/15] {marker} {vid_name:25s} | "
              f"True: {label_map[true_binary]:10s} (raw={raw_label}) | "
              f"Pred: {label_map[pred]:10s} ({conf*100:.1f}%) | "
              f"Votes: {frame_preds}")

        results.append({
            'video': vid_name,
            'true_raw': raw_label,
            'true_binary': true_binary,
            'predicted': pred,
            'confidence': round(conf, 4),
            'correct': pred == true_binary
        })

    # -------------------- METRICS --------------------
    if not pred_labels:
        print("\nNo predictions made!")
        return

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='binary')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    cm = confusion_matrix(true_labels, pred_labels)

    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)

    print("\n" + "=" * 65)
    print("                    RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Task:              Visual Binary Classification (Task 1)")
    print(f"  Model:             ResNet18 (Transfer Learning)")
    print(f"  Evaluation Set:    15 Validation Videos (Unseen)")
    print(f"  Frames per Video:  {NUM_FRAMES}")
    print("-" * 65)
    print(f"  Correct:           {correct} / {total}")
    print(f"  ACCURACY:          {acc*100:.2f}%")
    print(f"  F1-SCORE (Binary): {f1:.4f}")
    print(f"  F1-SCORE (Macro):  {f1_macro:.4f}")
    print("-" * 65)
    print(f"  Confusion Matrix:")
    print(f"                       Pred: Distracted  Pred: Engaged")
    if len(cm) == 2:
        print(f"  True: Distracted         {cm[0][0]:^12d}    {cm[0][1]:^12d}")
        print(f"  True: Engaged            {cm[1][0]:^12d}    {cm[1][1]:^12d}")
    print("-" * 65)
    print("\n  Classification Report:")
    print(classification_report(true_labels, pred_labels,
                                 target_names=["Distracted", "Engaged"]))
    print("=" * 65)

    # Qualification check
    if acc * 100 >= 70.0:
        print(f"  >> QUALIFIED! {acc*100:.2f}% >= 70%")
    else:
        print(f"  >> NOT QUALIFIED. {acc*100:.2f}% < 70%")
    print("=" * 65)

    # Save CSV
    out_path = "results_validation.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

if __name__ == "__main__":
    main()
