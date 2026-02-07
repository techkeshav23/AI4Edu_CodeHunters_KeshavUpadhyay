"""
Test script: Run trained LSTM model on a single video file.
Usage: python task1_2_seq/test_video.py --video path/to/video.avi --model task1_2_seq/model_seq.pth
"""
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import mediapipe as mp

# ========================= FEATURE EXTRACTION =========================
FPS_TARGET = 6
SEQ_LEN = 100

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
    if h_dist == 0: return 0
    return v_dist / h_dist

def extract_features(video_path):
    """Extract 9-dim features from video using MediaPipe Face Mesh."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(fps / FPS_TARGET)))

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f}, Total Frames: {total_frames}, Sampling every {frame_interval} frames")

    features = []
    frame_count = 0
    faces_found = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            results = face_mesh.process(image_rgb)

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
    face_mesh.close()

    features = np.array(features)
    print(f"  Extracted {len(features)} feature frames ({faces_found} with face detected)")
    return features

# ========================= MODEL (same as model_lstm.py) =========================
import torch.nn as nn

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
LABEL_MAP = {0: "0.0 (Distracted)", 1: "0.33 (Disengaged)",
             2: "0.66 (Nominally Engaged)", 3: "1.0 (Highly Engaged)"}

def bin_prediction(val):
    if val < 0.165: return 0
    elif val < 0.5: return 1
    elif val < 0.835: return 2
    else: return 3

def predict(features, model, device):
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

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Test trained LSTM model on a video")
    parser.add_argument('--video', type=str, required=True, help="Path to video file (e.g. 1.avi)")
    parser.add_argument('--model', type=str, default='task1_2_seq/model_seq.pth', help="Path to trained model")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = EngagementLSTM().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Model loaded: {args.model}")

    # Extract features
    print("\n--- Feature Extraction ---")
    features = extract_features(args.video)
    if features is None or len(features) == 0:
        print("ERROR: No features extracted from video!")
        sys.exit(1)

    # Predict
    print("\n--- Prediction ---")
    binary_pred, binary_prob, multi_class, reg_val = predict(features, model, device)

    print(f"\n{'='*50}")
    print(f"  Video: {os.path.basename(args.video)}")
    print(f"{'='*50}")
    print(f"  Task 1 (Binary):      {'1 (High Attentiveness)' if binary_pred == 1 else '0 (Low Attentiveness)'}  (prob: {binary_prob:.3f})")
    print(f"  Task 2 (Multi-class): {LABEL_MAP[multi_class]}  (raw: {reg_val:.3f})")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
