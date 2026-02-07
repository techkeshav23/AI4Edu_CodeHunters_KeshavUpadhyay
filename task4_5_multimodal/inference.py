"""
=========================================================
 Multimodal Inference Script (Task 4 & 5)
 
 Self-contained: extracts visual + rPPG features from video,
 runs fusion model, outputs predictions.
 
 Usage:
   # Single video
   python task4_5_multimodal/inference.py \
     --video path/to/video.avi \
     --rppg_results task3_rppg/results \
     --model task4_5_multimodal/fusion_best.pth
   
   # Batch (test set)
   python task4_5_multimodal/inference.py \
     --video_dir path/to/test/videos \
     --rppg_results task3_rppg/results \
     --model task4_5_multimodal/fusion_best.pth \
     --output results_task4_5.csv
=========================================================
"""

import os
import sys
import argparse
import gc
import csv
import numpy as np
import cv2
import torch
import mediapipe as mp
from scipy.signal import find_peaks, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalFusionModel

# ========================= CONFIG =========================
SEQ_LEN = 100
VISUAL_DIM = 9
RPPG_DIM = 14
FPS_TARGET = 6

VIDEO_EXTS = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm',
              '.AVI', '.MP4', '.MOV', '.MKV', '.WMV', '.WEBM')

CLASS_LABELS = {0: "Distracted", 1: "Disengaged", 2: "Nominally Engaged", 3: "Highly Engaged"}
BINARY_LABELS = {0: "Not Engaged", 1: "Engaged"}

# MediaPipe landmarks
POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]


# ========================= VISUAL FEATURES =========================

def get_head_pose(landmarks, shape):
    img_h, img_w, _ = shape
    face_2d, face_3d = [], []
    for idx, lm in enumerate(landmarks):
        if idx in POSE_LANDMARKS:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    return angles[0] * 360, angles[1] * 360, angles[2] * 360


def get_ear(landmarks, indices):
    top = landmarks[indices[2]]
    bottom = landmarks[indices[3]]
    left = landmarks[indices[0]]
    right = landmarks[indices[1]]
    vert = ((top.x - bottom.x)**2 + (top.y - bottom.y)**2)**0.5
    horiz = ((left.x - right.x)**2 + (left.y - right.y)**2)**0.5
    return vert / (horiz + 1e-6)


def extract_visual_features(video_path, face_mesh):
    """Extract 9-dim visual features from video at 6fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:
        fps = 30
    frame_interval = max(1, int(round(fps / FPS_TARGET)))
    
    features = []
    frame_count = 0
    MAX_FRAMES = 9000
    
    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            del image_rgb
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pitch, yaw, roll = get_head_pose(lm, frame.shape)
                le = lm[468] if len(lm) > 468 else lm[159]
                re = lm[473] if len(lm) > 473 else lm[386]
                l_ear = get_ear(lm, LEFT_EYE)
                r_ear = get_ear(lm, RIGHT_EYE)
                features.append([pitch, yaw, roll, le.x, le.y, re.x, re.y, l_ear, r_ear])
            else:
                features.append([0.0] * VISUAL_DIM)
        frame_count += 1
    
    cap.release()
    return np.array(features) if features else None


# ========================= rPPG FEATURES =========================

def extract_rppg_features_from_npy(npy_path):
    """Extract 14-dim rPPG feature from existing .npy result."""
    if not os.path.exists(npy_path):
        return np.zeros(RPPG_DIM, dtype=np.float32)
    
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception:
        return np.zeros(RPPG_DIM, dtype=np.float32)
    
    hr_pos = data.get('hr_pos', 0.0)
    hr_chrom = data.get('hr_chrom', 0.0)
    hr_deep = data.get('hr_deepphys', 0.0)
    sqi_pos = data.get('sqi_pos', 0.0)
    sqi_chrom = data.get('sqi_chrom', 0.0)
    sqi_deep = data.get('sqi_deepphys', 0.0)
    
    hrs = [hr_pos, hr_chrom, hr_deep]
    sqis = [sqi_pos, sqi_chrom, sqi_deep]
    hr_best = hrs[int(np.argmax(sqis))] if max(sqis) > 0 else np.mean(hrs)
    valid_hrs = [h for h in hrs if h > 0]
    hr_std = float(np.std(valid_hrs)) if len(valid_hrs) > 1 else 0.0
    
    # HRV from BVP signal
    fps = data.get('fps', 30.0)
    frames = data.get('frames', 1)
    eff_fps = 128.0 * fps / max(frames, 1) if frames > 128 else fps
    bvp = data.get('pos_signal', data.get('chrom_signal', None))
    
    mean_ibi, sdnn, rmssd = 0.0, 0.0, 0.0
    lf_power, hf_power, lf_hf = 0.0, 0.0, 0.0
    
    if bvp is not None and len(bvp) > 10:
        # HRV
        min_dist = max(1, int(0.4 * eff_fps))
        peaks, _ = find_peaks(bvp, distance=min_dist, height=0)
        if len(peaks) >= 3:
            ibis = np.diff(peaks) / eff_fps
            valid_ibis = ibis[(ibis > 0.33) & (ibis < 1.5)]
            if len(valid_ibis) >= 2:
                mean_ibi = float(np.mean(valid_ibis))
                sdnn = float(np.std(valid_ibis))
                sd = np.diff(valid_ibis)
                rmssd = float(np.sqrt(np.mean(sd**2))) if len(sd) > 0 else 0.0
        
        # Bandpower
        try:
            nperseg = min(64, len(bvp))
            freqs, psd = welch(bvp, fs=eff_fps, nperseg=nperseg, noverlap=nperseg//2)
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
            lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
            hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
            lf_hf = lf_power / (hf_power + 1e-10)
        except Exception:
            pass
    
    return np.array([
        hr_pos, hr_chrom, hr_deep,
        sqi_pos, sqi_chrom, sqi_deep,
        hr_best, hr_std,
        mean_ibi, sdnn, rmssd,
        lf_power, hf_power, lf_hf
    ], dtype=np.float32)


def normalize_rppg(rppg):
    """Normalize rPPG feature vector."""
    rppg_norm = rppg.copy()
    for i in [0, 1, 2, 6]:
        if i < len(rppg_norm):
            rppg_norm[i] /= 100.0
    if len(rppg_norm) > 7:
        rppg_norm[7] /= 50.0
    for i in [8, 9, 10]:
        if i < len(rppg_norm):
            rppg_norm[i] *= 10.0
    for i in [11, 12]:
        if i < len(rppg_norm):
            rppg_norm[i] = np.log1p(abs(rppg_norm[i]))
    if len(rppg_norm) > 13:
        rppg_norm[13] = min(rppg_norm[13] / 5.0, 1.0)
    return rppg_norm


# ========================= PREDICTION =========================

def predict(visual_feat, rppg_feat, model, device):
    """Run model inference."""
    # Normalize visual
    mean = np.mean(visual_feat, axis=0)
    std = np.std(visual_feat, axis=0) + 1e-6
    visual_norm = (visual_feat - mean) / std
    
    # Pad/subsample
    L, D = visual_norm.shape
    if L < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - L, D))
        visual_norm = np.vstack([visual_norm, pad])
    elif L > SEQ_LEN:
        indices = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
        visual_norm = visual_norm[indices]
    
    # Normalize rPPG
    rppg_norm = normalize_rppg(rppg_feat)
    
    visual_t = torch.FloatTensor(visual_norm).unsqueeze(0).to(device)
    rppg_t = torch.FloatTensor(rppg_norm).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_bin, out_multi, out_reg = model(visual_t, rppg_t)
    
    bin_prob = torch.sigmoid(out_bin).item()
    bin_pred = 1 if bin_prob > 0.5 else 0
    
    cls_probs = torch.softmax(out_multi, dim=1).cpu().numpy()[0]
    cls_pred = int(np.argmax(cls_probs))
    
    return bin_pred, bin_prob, cls_pred, cls_probs


def main():
    parser = argparse.ArgumentParser(description="Task 4/5 Multimodal Inference")
    parser.add_argument('--video', type=str, default=None, help='Single video path')
    parser.add_argument('--video_dir', type=str, default=None, help='Directory of test videos')
    parser.add_argument('--rppg_results', type=str, default='task3_rppg/results',
                        help='Dir with rPPG .npy results')
    parser.add_argument('--model', type=str, default='task4_5_multimodal/fusion_best.pth',
                        help='Path to trained fusion model')
    parser.add_argument('--output', type=str, default='results_task4_5.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = MultimodalFusionModel(
        visual_dim=VISUAL_DIM, rppg_dim=RPPG_DIM,
        lstm_hidden=32, lstm_layers=2, dropout=0.3
    ).to(device)
    
    if os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model: {args.model}")
    else:
        print(f"WARNING: Model not found: {args.model}")
    
    model.eval()
    
    # Collect videos
    videos = []
    if args.video:
        videos.append(args.video)
    elif args.video_dir:
        for f in sorted(os.listdir(args.video_dir)):
            if any(f.endswith(ext) for ext in VIDEO_EXTS):
                videos.append(os.path.join(args.video_dir, f))
    
    if not videos:
        print("No videos found!")
        return
    
    print(f"Processing {len(videos)} videos...")
    
    # Initialize MediaPipe once (memory-safe singleton)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    results = []
    
    for i, vid_path in enumerate(videos):
        name = os.path.splitext(os.path.basename(vid_path))[0]
        print(f"  [{i+1}/{len(videos)}] {name}...", end=' ', flush=True)
        
        try:
            # 1. Extract visual features
            visual = extract_visual_features(vid_path, face_mesh)
            if visual is None or len(visual) < 5:
                print("SKIP (no face)")
                results.append([name, -1, 0.0, -1, "N/A", "N/A"])
                continue
            
            # 2. Get rPPG features
            rppg_path = os.path.join(args.rppg_results, name + '_rppg.npy')
            rppg = extract_rppg_features_from_npy(rppg_path)
            
            # 3. Predict
            bin_pred, bin_prob, cls_pred, cls_probs = predict(visual, rppg, model, device)
            
            print(f"Binary: {BINARY_LABELS[bin_pred]} ({bin_prob:.2f}) | "
                  f"Class: {CLASS_LABELS[cls_pred]} ({cls_probs[cls_pred]:.2f})")
            
            results.append([name, bin_pred, bin_prob, cls_pred,
                           BINARY_LABELS[bin_pred], CLASS_LABELS[cls_pred]])
            
            # Cleanup
            del visual, rppg
            gc.collect()
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append([name, -1, 0.0, -1, "Error", str(e)])
    
    face_mesh.close()
    
    # Save results
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'binary_pred', 'binary_prob', 'class_pred',
                         'binary_label', 'class_label'])
        writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
