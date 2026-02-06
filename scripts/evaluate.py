"""
=========================================================
 Student Engagement Recognition - Evaluation Script
 Task 1: Visual Binary Classification
 
 This script is designed for judges/evaluators to test
 the model on unseen test videos and get metrics.
 
 Usage:
   python scripts/evaluate.py --test_dir <path_to_test_videos> --labels <path_to_labels.xlsx> --model models/visual/best_visual_model.pth
   
 Output:
   - Per-video predictions (printed + saved to CSV)
   - Accuracy, F1-Score, Confusion Matrix
=========================================================
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import mediapipe as mp

# -------------------- MODEL --------------------
class VisualEngagementModel(torch.nn.Module):
    """ResNet18-based binary engagement classifier."""
    def __init__(self, num_classes=2):
        super(VisualEngagementModel, self).__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# -------------------- FACE DETECTOR --------------------
class FaceDetector:
    """MediaPipe face detector matching training pipeline."""
    def __init__(self, confidence=0.5, padding=0):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=confidence
        )
        self.padding = padding  # Must match training (0)

    def extract_face(self, frame):
        """Returns cropped face from a single frame, or None."""
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

# -------------------- PREDICTION --------------------
def predict_video(video_path, model, face_detector, transform, device, num_frames=10):
    """
    Predict engagement for a single video.
    
    Strategy: Sample multiple frames from the video, predict each,
    and use majority voting for the final prediction.
    
    Args:
        video_path: Path to video file
        model: Loaded PyTorch model
        face_detector: FaceDetector instance 
        transform: Image transforms
        device: torch device
        num_frames: Number of frames to sample from video
        
    Returns:
        predicted_class (int), confidence (float), per_frame_preds (list)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return -1, 0.0, []
    
    # Sample frames evenly across the video
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
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
        
        # Preprocess
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(face_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
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
    
    # Majority voting
    from collections import Counter
    vote = Counter(predictions).most_common(1)[0][0]
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == vote])
    
    return vote, avg_conf, predictions

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Visual Engagement Model on Test Videos"
    )
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to folder containing test videos")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to Excel/CSV with test labels (col0=name, col1=label)")
    parser.add_argument("--model", type=str, default="models/visual/best_visual_model.pth",
                        help="Path to trained model (.pth)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames to sample per video (default: 10)")
    parser.add_argument("--output", type=str, default="results_task1.csv",
                        help="Output CSV filename for predictions")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to evaluate (0 = all)")
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    print(f"Loading model: {args.model}")
    model = VisualEngagementModel(num_classes=2)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Face Detector (same settings as training)
    face_detector = FaceDetector(confidence=0.5, padding=0)
    
    # Transforms (same as validation transforms used in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load Labels
    if args.labels.endswith('.csv'):
        df = pd.read_csv(args.labels)
    else:
        df = pd.read_excel(args.labels)
    
    video_names = df.iloc[:, 0].astype(str).tolist()
    raw_labels = df.iloc[:, 1].tolist()
    
    # Convert to binary labels (matching training)
    # 0, 0.33 -> 0 (Low/Distracted)  |  0.66, 1 -> 1 (High/Engaged)
    true_labels = [0 if lbl <= 0.33 else 1 for lbl in raw_labels]
    
    # Limit videos if --max_videos is set
    if args.max_videos > 0:
        video_names = video_names[:args.max_videos]
        raw_labels = raw_labels[:args.max_videos]
        true_labels = true_labels[:args.max_videos]
    
    print(f"Test videos: {len(video_names)}")
    print(f"Test directory: {args.test_dir}")
    print("=" * 60)
    
    # Supported video extensions
    EXTENSIONS = ['.mp4', '.MP4', '.avi', '.AVI', '.webm', '.wmv', '.mov']
    
    # Run Predictions
    predicted_labels = []
    results_data = []
    skipped = 0
    
    for i, (name, true_lbl, raw_lbl) in enumerate(zip(video_names, true_labels, raw_labels)):
        # Find the video file (handle extension issues)
        video_path = None
        base_name = os.path.splitext(name)[0]
        
        # Try exact name first
        exact_path = os.path.join(args.test_dir, name)
        if os.path.isfile(exact_path):
            video_path = exact_path
        else:
            # Try all extensions
            for ext in EXTENSIONS:
                test_path = os.path.join(args.test_dir, base_name + ext)
                if os.path.isfile(test_path):
                    video_path = test_path
                    break
        
        if video_path is None:
            print(f"  [{i+1}/{len(video_names)}] SKIP - Video not found: {name}")
            skipped += 1
            continue
        
        # Predict
        pred, conf, frame_preds = predict_video(
            video_path, model, face_detector, transform, device, args.num_frames
        )
        
        if pred == -1:
            print(f"  [{i+1}/{len(video_names)}] FAIL - No faces detected: {name}")
            skipped += 1
            continue
        
        predicted_labels.append(pred)
        
        status = "✓" if pred == true_lbl else "✗"
        label_map = {0: "Distracted", 1: "Engaged"}
        print(f"  [{i+1}/{len(video_names)}] {status} {name:30s} | "
              f"True: {label_map[true_lbl]:10s} (raw={raw_lbl}) | "
              f"Pred: {label_map[pred]:10s} ({conf*100:.1f}%) | "
              f"Frames: {frame_preds}")
        
        results_data.append({
            'video': name,
            'true_raw_label': raw_lbl,
            'true_binary_label': true_lbl,
            'predicted_label': pred,
            'confidence': round(conf, 4),
            'true_class': label_map[true_lbl],
            'predicted_class': label_map[pred],
            'correct': pred == true_lbl
        })
    
    # -------------------- METRICS --------------------
    if not predicted_labels:
        print("\nNo predictions were made. Check video paths.")
        return
    
    true_for_eval = [r['true_binary_label'] for r in results_data]
    pred_for_eval = [r['predicted_label'] for r in results_data]
    
    acc = accuracy_score(true_for_eval, pred_for_eval)
    f1 = f1_score(true_for_eval, pred_for_eval, average='binary')
    f1_macro = f1_score(true_for_eval, pred_for_eval, average='macro')
    cm = confusion_matrix(true_for_eval, pred_for_eval)
    
    print("\n" + "=" * 60)
    print("               EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Task:           Visual Binary Classification (Task 1)")
    print(f"  Model:          ResNet18 (Transfer Learning)")
    print(f"  Videos Tested:  {len(results_data)}")
    print(f"  Videos Skipped: {skipped}")
    print(f"  Frames/Video:   {args.num_frames}")
    print("-" * 60)
    print(f"  ACCURACY:       {acc*100:.2f}%")
    print(f"  F1-SCORE:       {f1:.4f}")
    print(f"  F1 (Macro):     {f1_macro:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"                    Pred: Distracted  Pred: Engaged")
    print(f"  True: Distracted      {cm[0][0]:^12d}    {cm[0][1]:^12d}")
    print(f"  True: Engaged         {cm[1][0]:^12d}    {cm[1][1]:^12d}")
    print("-" * 60)
    print("\n  Classification Report:")
    print(classification_report(true_for_eval, pred_for_eval, 
                                 target_names=["Distracted", "Engaged"]))
    print("=" * 60)
    
    # Qualification Check
    QUALIFICATION_THRESHOLD = 70.0
    if acc * 100 >= QUALIFICATION_THRESHOLD:
        print(f"  >> QUALIFIED! Accuracy {acc*100:.2f}% >= {QUALIFICATION_THRESHOLD}%")
    else:
        print(f"  >> NOT QUALIFIED. Accuracy {acc*100:.2f}% < {QUALIFICATION_THRESHOLD}%")
    
    # Save Results CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(args.output, index=False)
    print(f"\n  Results saved to: {args.output}")

if __name__ == "__main__":
    main()
