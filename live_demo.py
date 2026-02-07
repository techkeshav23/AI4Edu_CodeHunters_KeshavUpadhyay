"""
=========================================================
 LIVE DEMO — Video + Real-time Engagement Graphs
 
 Shows the video playing with face mesh overlay, and
 live-updating graphs for engagement prediction.
 
 Layout (single window):
   LEFT:  Video feed with face landmarks + engagement label
   RIGHT: Live graphs (engagement score, head pose, EAR)
 
 Usage:
   python live_demo.py --video 1.avi
   python live_demo.py --video 1.avi --model task1_2_seq/model_seq.pth
   python live_demo.py --webcam          (for webcam demo)
=========================================================
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from collections import deque

# ========================= CONFIG =========================
FPS_TARGET = 6
SEQ_LEN = 100
FEATURE_DIM = 9
GRAPH_HISTORY = 120  # frames of history to display

# Colors (BGR)
COLOR_BG = (30, 30, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 220, 100)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_BLUE = (255, 180, 0)
COLOR_ORANGE = (0, 140, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_PURPLE = (200, 100, 255)
COLOR_GRID = (60, 60, 60)

# Engagement class labels
LABELS = {0: "Distracted", 1: "Disengaged", 2: "Nominally Engaged", 3: "Highly Engaged"}
LABEL_COLORS = {
    0: COLOR_RED, 1: COLOR_ORANGE, 2: COLOR_YELLOW, 3: COLOR_GREEN
}
LABEL_VALUES = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}


# ========================= FEATURE EXTRACTION =========================
def get_head_pose(landmarks, shape):
    img_h, img_w, _ = shape
    face_2d, face_3d = [], []
    key_landmarks = [1, 199, 33, 263, 61, 291]
    for idx, lm in enumerate(landmarks):
        if idx in key_landmarks:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return 0, 0, 0
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles = cv2.RQDecomp3x3(rmat)[0]
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


def extract_frame(lm, shape):
    """Extract 9-dim feature from one frame."""
    pitch, yaw, roll = get_head_pose(lm, shape)
    le_x = (lm[33].x + lm[133].x) / 2
    le_y = (lm[33].y + lm[133].y) / 2
    re_x = (lm[362].x + lm[263].x) / 2
    re_y = (lm[362].y + lm[263].y) / 2
    left_ear = get_eye_ratio(lm, [33, 159, 133, 145])
    right_ear = get_eye_ratio(lm, [362, 386, 263, 374])
    return [pitch, yaw, roll, le_x, le_y, re_x, re_y, left_ear, right_ear]


# ========================= MODEL =========================
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1), weights

class EngagementLSTM(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 32), torch.nn.ReLU(), torch.nn.Dropout(0.3))
        self.head_binary = torch.nn.Linear(32, 1)
        self.head_regression = torch.nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, _ = self.attention(out)
        feat = self.fc(ctx)
        return self.head_binary(feat), self.head_regression(feat)


def regression_to_class(val):
    if val < 0.165: return 0
    if val < 0.5: return 1
    if val < 0.835: return 2
    return 3


# ========================= GRAPH DRAWING =========================
def draw_graph(canvas, data, x_start, y_start, width, height, 
               title, color, y_min=0, y_max=1, show_grid=True,
               threshold=None, threshold_color=None):
    """Draw a line graph on the canvas."""
    # Background
    cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height),
                  (40, 40, 40), -1)
    cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height),
                  (80, 80, 80), 1)
    
    # Title
    cv2.putText(canvas, title, (x_start + 8, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Grid lines
    if show_grid:
        for i in range(5):
            gy = y_start + int(height * i / 4)
            cv2.line(canvas, (x_start, gy), (x_start + width, gy), COLOR_GRID, 1)
            val = y_max - (y_max - y_min) * i / 4
            cv2.putText(canvas, f"{val:.1f}", (x_start + 2, gy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    
    # Threshold line
    if threshold is not None:
        th_y = y_start + int(height * (1 - (threshold - y_min) / (y_max - y_min + 1e-6)))
        th_y = max(y_start, min(y_start + height, th_y))
        cv2.line(canvas, (x_start, th_y), (x_start + width, th_y),
                 threshold_color or COLOR_YELLOW, 1, cv2.LINE_AA)
    
    # Plot data
    if len(data) < 2:
        return
    
    margin_top = 25
    plot_h = height - margin_top - 5
    plot_y = y_start + margin_top
    
    points = []
    for i, val in enumerate(data):
        px = x_start + int(width * i / max(len(data) - 1, 1))
        normalized = (val - y_min) / (y_max - y_min + 1e-6)
        normalized = max(0, min(1, normalized))
        py = plot_y + int(plot_h * (1 - normalized))
        points.append((px, py))
    
    for i in range(1, len(points)):
        cv2.line(canvas, points[i-1], points[i], color, 2, cv2.LINE_AA)
    
    # Current value
    if data:
        cv2.putText(canvas, f"{data[-1]:.2f}", (x_start + width - 55, y_start + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_bar_chart(canvas, labels_list, values, colors, x_start, y_start, width, height, title):
    """Draw a horizontal bar chart for class probabilities."""
    cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height),
                  (40, 40, 40), -1)
    cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height),
                  (80, 80, 80), 1)
    cv2.putText(canvas, title, (x_start + 8, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    n = len(labels_list)
    bar_h = max(12, (height - 30) // n - 4)
    bar_max_w = width - 120
    
    for i, (label, val, col) in enumerate(zip(labels_list, values, colors)):
        by = y_start + 28 + i * (bar_h + 4)
        bw = int(bar_max_w * max(0, min(1, val)))
        
        # Bar background
        cv2.rectangle(canvas, (x_start + 100, by), (x_start + 100 + bar_max_w, by + bar_h),
                      (50, 50, 50), -1)
        # Bar fill
        if bw > 0:
            cv2.rectangle(canvas, (x_start + 100, by), (x_start + 100 + bw, by + bar_h),
                          col, -1)
        # Label
        cv2.putText(canvas, label[:12], (x_start + 5, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)
        # Value
        cv2.putText(canvas, f"{val:.2f}", (x_start + 105 + bar_max_w, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)


def draw_face_mesh_light(frame, landmarks, engaged_color):
    """Draw key face landmarks on frame (lightweight)."""
    h, w = frame.shape[:2]
    
    # Face oval (selected points)
    oval_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    pts = []
    for idx in oval_idx:
        if idx < len(landmarks):
            pts.append((int(landmarks[idx].x * w), int(landmarks[idx].y * h)))
    if len(pts) > 2:
        cv2.polylines(frame, [np.array(pts)], True, engaged_color, 1, cv2.LINE_AA)
    
    # Eyes
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    for eye_idx in [left_eye, right_eye]:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx if i < len(landmarks)]
        if pts:
            cv2.polylines(frame, [np.array(pts)], True, engaged_color, 1, cv2.LINE_AA)
    
    # Iris dots
    iris_idx = [468, 473]  # left, right iris center
    for idx in iris_idx:
        if idx < len(landmarks):
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), 3, COLOR_CYAN, -1)


# ========================= MAIN DEMO =========================
def run_demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = EngagementLSTM().to(device)
    if args.model and os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Model loaded: {args.model}")
    else:
        print("WARNING: No model loaded — predictions will be random!")
    model.eval()
    
    # Video source
    if args.webcam:
        cap = cv2.VideoCapture(0)
        video_name = "Webcam"
    else:
        cap = cv2.VideoCapture(args.video)
        video_name = os.path.basename(args.video)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open {'webcam' if args.webcam else args.video}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(fps / FPS_TARGET)))
    
    # MediaPipe
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    # History buffers for graphs
    engagement_history = deque(maxlen=GRAPH_HISTORY)
    pitch_history = deque(maxlen=GRAPH_HISTORY)
    yaw_history = deque(maxlen=GRAPH_HISTORY)
    ear_history = deque(maxlen=GRAPH_HISTORY)
    
    # Feature buffer for model (sliding window)
    feature_buffer = deque(maxlen=SEQ_LEN)
    
    # State
    current_class = 1
    current_prob = 0.5
    current_reg = 0.33
    frame_count = 0
    processed = 0
    face_found = False
    
    # Window config
    VIDEO_W, VIDEO_H = 640, 480
    GRAPH_W = 420
    TOTAL_W = VIDEO_W + GRAPH_W + 20
    TOTAL_H = max(VIDEO_H + 80, 560)
    
    window_name = f"ADVITIYA Live Demo - {video_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TOTAL_W, TOTAL_H)
    
    print(f"\n{'='*60}")
    print(f"  LIVE DEMO — {video_name}")
    print(f"  FPS: {fps:.0f} | Frames: {total_frames} | Sampling: 1/{frame_interval}")
    print(f"  Press 'Q' to quit, 'P' to pause, 'R' to restart")
    print(f"{'='*60}\n")
    
    paused = False
    start_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not args.webcam:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                else:
                    break
            
            frame_count += 1
            
            # Resize video frame
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            
            # Process at target FPS
            if frame_count % frame_interval == 0:
                processed += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    face_found = True
                    lm = results.multi_face_landmarks[0].landmark
                    feat = extract_frame(lm, (VIDEO_H, VIDEO_W, 3))
                    feature_buffer.append(feat)
                    
                    # Update history for graphs
                    pitch_history.append(feat[0])
                    yaw_history.append(feat[1])
                    ear_history.append((feat[7] + feat[8]) / 2)
                    
                    # Draw face mesh
                    eng_color = LABEL_COLORS.get(current_class, COLOR_WHITE)
                    draw_face_mesh_light(frame, lm, eng_color)
                    
                    # Run model prediction (if enough frames)
                    if len(feature_buffer) >= 10:
                        features = np.array(list(feature_buffer))
                        # Normalize
                        mean = np.mean(features, axis=0)
                        std = np.std(features, axis=0) + 1e-6
                        features_norm = (features - mean) / std
                        
                        # Pad to SEQ_LEN
                        L = features_norm.shape[0]
                        if L < SEQ_LEN:
                            pad = np.zeros((SEQ_LEN - L, FEATURE_DIM))
                            features_norm = np.vstack([features_norm, pad])
                        elif L > SEQ_LEN:
                            indices = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
                            features_norm = features_norm[indices]
                        
                        feat_t = torch.FloatTensor(features_norm).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            out_bin, out_reg = model(feat_t)
                            current_prob = torch.sigmoid(out_bin).item()
                            current_reg = max(0, min(1, out_reg.item()))
                            current_class = regression_to_class(current_reg)
                        
                        engagement_history.append(current_reg)
                else:
                    face_found = False
                    feature_buffer.append([0] * FEATURE_DIM)
                    engagement_history.append(0)
                    pitch_history.append(0)
                    yaw_history.append(0)
                    ear_history.append(0)
        
        # ==================== BUILD DISPLAY ====================
        canvas = np.full((TOTAL_H, TOTAL_W, 3), 30, dtype=np.uint8)
        
        # --- LEFT: Video ---
        canvas[40:40+VIDEO_H, 10:10+VIDEO_W] = frame
        
        # Video border color based on engagement
        border_color = LABEL_COLORS.get(current_class, COLOR_WHITE)
        cv2.rectangle(canvas, (8, 38), (12+VIDEO_W, 42+VIDEO_H), border_color, 2)
        
        # Top bar — title
        cv2.putText(canvas, f"ADVITIYA - Student Engagement Analysis", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        # Engagement label on video
        label_text = LABELS[current_class]
        label_color = LABEL_COLORS[current_class]
        
        # Label background
        label_y = 40 + VIDEO_H + 5
        cv2.rectangle(canvas, (10, label_y), (10 + VIDEO_W, label_y + 35), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, label_y), (10 + VIDEO_W, label_y + 35), border_color, 2)
        
        # Binary result
        binary_text = "HIGH ATTENTIVENESS" if current_prob > 0.5 else "LOW ATTENTIVENESS"
        binary_color = COLOR_GREEN if current_prob > 0.5 else COLOR_RED
        cv2.putText(canvas, f"Task1: {binary_text} ({current_prob:.2f})",
                    (18, label_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, binary_color, 1)
        
        # Multi-class result
        cv2.putText(canvas, f"Task2: {label_text} ({current_reg:.2f})",
                    (18, label_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
        
        # Face status
        face_status = "FACE DETECTED" if face_found else "NO FACE"
        face_color = COLOR_GREEN if face_found else COLOR_RED
        cv2.putText(canvas, face_status, (VIDEO_W - 130, label_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_color, 1)
        
        # Frame counter
        elapsed = time.time() - start_time
        cv2.putText(canvas, f"Frame: {frame_count}/{total_frames}  Time: {elapsed:.1f}s",
                    (VIDEO_W - 230, label_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # --- RIGHT: Graphs ---
        gx = VIDEO_W + 25  # graph x start
        gw = GRAPH_W - 20  # graph width
        
        # 1. Engagement Score (main graph — tall)
        draw_graph(canvas, list(engagement_history),
                   gx, 40, gw, 130,
                   "Engagement Score", COLOR_GREEN,
                   y_min=0, y_max=1.0,
                   threshold=0.33, threshold_color=COLOR_YELLOW)
        
        # 2. Class probability bars
        class_names = ["Not Engaged", "Barely Eng.", "Engaged", "Highly Eng."]
        class_colors = [COLOR_RED, COLOR_ORANGE, COLOR_YELLOW, COLOR_GREEN]
        # Approximate softmax-like distribution from regression
        centers = np.array([0.0, 0.33, 0.66, 1.0])
        distances = np.abs(current_reg - centers)
        probs = np.exp(-distances / 0.12)
        probs = probs / (probs.sum() + 1e-8)
        
        draw_bar_chart(canvas, class_names, probs.tolist(), class_colors,
                       gx, 180, gw, 100, "Class Distribution")
        
        # 3. Head Pitch (looking up/down)
        draw_graph(canvas, list(pitch_history),
                   gx, 290, gw, 80,
                   "Head Pitch (Up/Down)", COLOR_BLUE,
                   y_min=-30, y_max=30)
        
        # 4. Head Yaw (looking left/right)
        draw_graph(canvas, list(yaw_history),
                   gx, 380, gw, 80,
                   "Head Yaw (Left/Right)", COLOR_PURPLE,
                   y_min=-40, y_max=40)
        
        # 5. Eye Aspect Ratio (blink/drowsiness)
        draw_graph(canvas, list(ear_history),
                   gx, 470, gw, 80,
                   "Eye Openness (EAR)", COLOR_CYAN,
                   y_min=0, y_max=0.5,
                   threshold=0.15, threshold_color=COLOR_RED)
        
        # Pause indicator
        if paused:
            cv2.putText(canvas, "|| PAUSED", (TOTAL_W // 2 - 60, TOTAL_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)
        
        # Controls help
        cv2.putText(canvas, "Q:Quit  P:Pause  R:Restart", (gx, TOTAL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        # Show
        cv2.imshow(window_name, canvas)
        
        # Playback speed control
        if args.webcam:
            delay = 1
        else:
            delay = max(1, int(1000 / fps))
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            feature_buffer.clear()
            engagement_history.clear()
            pitch_history.clear()
            yaw_history.clear()
            ear_history.clear()
            start_time = time.time()
    
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


def main():
    parser = argparse.ArgumentParser(description="Live Demo — Video + Engagement Graphs")
    parser.add_argument('--video', type=str, default='1.avi', help='Video file path')
    parser.add_argument('--model', type=str, default='task1_2_seq/model_seq.pth',
                        help='Path to trained model weights')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of video file')
    args = parser.parse_args()
    
    if not args.webcam and not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    run_demo(args)


if __name__ == "__main__":
    main()
