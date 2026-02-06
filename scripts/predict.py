"""
=========================================================
 Student Engagement Recognition - Single Video Prediction
 Task 1: Visual Binary Classification
 
 This script predicts engagement for a single test video.
 No labels needed - just give a video and get prediction.
 
 Usage:
   python scripts/predict.py --video <path_to_video> --model models/visual/best_visual_model.pth
=========================================================
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
from torchvision import transforms, models
from collections import Counter
import mediapipe as mp
from PIL import Image

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

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(description="Predict engagement for a single video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="models/visual/best_visual_model.pth",
                        help="Path to trained model (.pth)")
    parser.add_argument("--num_frames", type=int, default=15,
                        help="Number of frames to sample (default: 15)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = VisualEngagementModel(num_classes=2)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Face Detector
    face_detector = FaceDetector(confidence=0.5, padding=0)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Process Video
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo: {os.path.basename(args.video)}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.0f} | Duration: {duration:.1f}s")
    print("-" * 50)
    
    # Sample frames
    if total_frames <= args.num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int).tolist()
    
    predictions = []
    confidences = []
    label_map = {0: "Distracted", 1: "Engaged"}
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        face = face_detector.extract_face(frame)
        if face is None:
            print(f"  Frame {idx:5d}: No face detected")
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
        print(f"  Frame {idx:5d}: {label_map[pred]:12s} (confidence: {conf*100:.1f}%)")
    
    cap.release()
    
    if not predictions:
        print("\nNo faces detected in any frame!")
        return
    
    # Final Result (Majority Voting)
    vote = Counter(predictions).most_common(1)[0][0]
    engaged_count = predictions.count(1)
    distracted_count = predictions.count(0)
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == vote])
    
    print("\n" + "=" * 50)
    print(f"  FINAL PREDICTION: {label_map[vote].upper()}")
    print(f"  Confidence:       {avg_conf*100:.1f}%")
    print(f"  Engaged Frames:   {engaged_count}/{len(predictions)}")
    print(f"  Distracted Frames:{distracted_count}/{len(predictions)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
