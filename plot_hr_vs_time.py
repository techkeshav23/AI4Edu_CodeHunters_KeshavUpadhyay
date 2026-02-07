"""
=========================================================
 rPPG: Heart Rate vs Time Graph
 
 Processes a video → extracts rPPG signal → windowed HR
 Shows how heart rate changes over the video duration.
 
 Usage:
   python plot_hr_vs_time.py --video 1.avi
   python plot_hr_vs_time.py --video data/raw/videos/Train/subject_10_Vid_6.avi
=========================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib

# Import DeepPhys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task3_rppg'))
try:
    from deepphys import DeepPhysExtractor
    HAS_DEEPPHYS = True
except ImportError:
    HAS_DEEPPHYS = False
    print("  WARNING: DeepPhys not available, skipping CNN-based rPPG")

# ========================= CONFIG =========================
FREQ_LOW = 0.7    # 42 BPM
FREQ_HIGH = 3.0   # 180 BPM
FILTER_ORDER = 4
WINDOW_SEC = 10    # 10-second sliding window
STEP_SEC = 1       # 1-second step

# MediaPipe face ROI landmarks
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132,
                93, 234, 127, 162, 21, 54, 103, 67, 109]

LEFT_CHEEK_IDX = [116, 117, 118, 119, 100, 36, 205, 187, 123, 
                  50, 101, 102, 48, 64, 235]

RIGHT_CHEEK_IDX = [345, 346, 347, 348, 329, 266, 425, 411, 352,
                   280, 330, 331, 278, 294, 455]

ALL_ROI_IDX = FOREHEAD_IDX + LEFT_CHEEK_IDX + RIGHT_CHEEK_IDX

# ========================= FUNCTIONS =========================
def get_roi_mean_rgb(frame, landmarks, roi_indices, img_w, img_h):
    points = []
    for idx in roi_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            points.append([x, y])
    if len(points) < 3:
        return None
    points = np.array(points, dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    mean_rgb = cv2.mean(frame, mask=mask)[:3]
    return mean_rgb  # BGR

def bandpass_filter(signal, fs, low=FREQ_LOW, high=FREQ_HIGH, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))

def pos_algorithm(rgb_signals):
    """POS (Plane Orthogonal to Skin) algorithm."""
    rgb = np.array(rgb_signals)  # (N, 3)
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    # Temporal normalization
    mean_rgb = np.mean(rgb, axis=0, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1
    Cn = rgb / mean_rgb
    # POS projection
    S1 = Cn[:, 1] - Cn[:, 2]  # G - B
    S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R
    # Adaptive ratio
    std1 = np.std(S1)
    std2 = np.std(S2)
    if std2 == 0:
        return S1
    alpha = std1 / std2
    pulse = S1 + alpha * S2
    return pulse

def chrom_algorithm(rgb_signals):
    """CHROM (Chrominance-based) algorithm."""
    rgb = np.array(rgb_signals)
    if len(rgb) < 10:
        return np.zeros(len(rgb))
    mean_rgb = np.mean(rgb, axis=0, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1
    Cn = rgb / mean_rgb
    Xs = 3 * Cn[:, 0] - 2 * Cn[:, 1]   # 3R - 2G
    Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]  # 1.5R + G - 1.5B
    std_x = np.std(Xs)
    std_y = np.std(Ys)
    if std_y == 0:
        return Xs
    alpha = std_x / std_y
    pulse = Xs - alpha * Ys
    return pulse

def estimate_hr_from_window(signal, fs):
    """Estimate HR from a short signal window using FFT."""
    if len(signal) < fs * 2:  # Need at least 2 seconds
        return None
    try:
        filtered = bandpass_filter(signal, fs)
        freqs, psd = welch(filtered, fs=fs, nperseg=min(256, len(filtered)), 
                          noverlap=min(128, len(filtered)//2))
        valid = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
        if not np.any(valid):
            return None
        peak_freq = freqs[valid][np.argmax(psd[valid])]
        hr = peak_freq * 60
        return hr
    except:
        return None

# ========================= MAIN =========================
def process_video(video_path):
    print(f"\nProcessing: {video_path}")
    
    # Video hash for watermark
    with open(video_path, 'rb') as f:
        vid_hash = hashlib.md5(f.read(1024*64)).hexdigest()[:12]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration:.1f}s")
    
    # Extract RGB signals
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    rgb_signals = []
    face_crops = []  # For DeepPhys
    frame_idx = 0
    
    # Haar cascade for DeepPhys face crops
    haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print("  Extracting face ROI RGB signals + face crops...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mean_bgr = get_roi_mean_rgb(frame, landmarks, ALL_ROI_IDX, img_w, img_h)
            if mean_bgr is not None:
                rgb_signals.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])  # RGB
            else:
                if rgb_signals:
                    rgb_signals.append(rgb_signals[-1])
                else:
                    rgb_signals.append([0, 0, 0])
            # Collect face crop for DeepPhys
            face_crops.append(frame.copy())
        else:
            if rgb_signals:
                rgb_signals.append(rgb_signals[-1])
            else:
                rgb_signals.append([0, 0, 0])
            face_crops.append(frame.copy())
        
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"    Frame {frame_idx}/{total_frames}")
    
    cap.release()
    face_mesh.close()
    
    rgb_signals = np.array(rgb_signals)  # (N, 3)
    print(f"  Extracted {len(rgb_signals)} frames of RGB data")
    
    # Full signal rPPG
    pos_signal = pos_algorithm(rgb_signals)
    chrom_signal = chrom_algorithm(rgb_signals)
    pos_filtered = bandpass_filter(pos_signal, fps)
    chrom_filtered = bandpass_filter(chrom_signal, fps)
    
    # DeepPhys BVP extraction
    deep_signal = None
    deep_filtered = None
    if HAS_DEEPPHYS and len(face_crops) > int(fps * 5):
        print("  Running DeepPhys CNN on face crops...")
        try:
            extractor = DeepPhysExtractor()
            mode = "trained" if extractor.pretrained else "UNTRAINED"
            print(f"    DeepPhys mode: {mode} | Device: {extractor.device}")
            hr_deep, sqi_deep, bvp_deep = extractor.extract_hr_from_frames(face_crops, fps)
            if bvp_deep is not None and len(bvp_deep) > 0:
                deep_signal = bvp_deep
                deep_filtered = bandpass_filter(deep_signal, fps)
                print(f"    DeepPhys BVP: {len(deep_signal)} frames, overall HR: {hr_deep:.1f} BPM")
            else:
                print("    DeepPhys: insufficient signal")
        except Exception as e:
            print(f"    DeepPhys error: {e}")
    elif not HAS_DEEPPHYS:
        print("  Skipping DeepPhys (not available)")
    
    # Free face crops memory
    del face_crops
    import gc; gc.collect()
    
    # Windowed HR estimation
    window_frames = int(WINDOW_SEC * fps)
    step_frames = int(STEP_SEC * fps)
    
    hr_times_pos = []
    hr_values_pos = []
    hr_times_chrom = []
    hr_values_chrom = []
    hr_times_deep = []
    hr_values_deep = []
    
    print(f"  Computing windowed HR (window={WINDOW_SEC}s, step={STEP_SEC}s)...")
    
    for start in range(0, len(pos_signal) - window_frames, step_frames):
        end = start + window_frames
        t_center = (start + end) / 2 / fps
        
        hr_p = estimate_hr_from_window(pos_signal[start:end], fps)
        hr_c = estimate_hr_from_window(chrom_signal[start:end], fps)
        
        if hr_p is not None and 40 <= hr_p <= 180:
            hr_times_pos.append(t_center)
            hr_values_pos.append(hr_p)
        
        if hr_c is not None and 40 <= hr_c <= 180:
            hr_times_chrom.append(t_center)
            hr_values_chrom.append(hr_c)
    
    # DeepPhys windowed HR (signal may be shorter due to frame diffs)
    if deep_signal is not None and len(deep_signal) > window_frames:
        for start in range(0, len(deep_signal) - window_frames, step_frames):
            end = start + window_frames
            t_center = (start + end) / 2 / fps
            hr_d = estimate_hr_from_window(deep_signal[start:end], fps)
            if hr_d is not None and 40 <= hr_d <= 180:
                hr_times_deep.append(t_center)
                hr_values_deep.append(hr_d)
    
    # ========================= PLOT =========================
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_axis = np.arange(len(rgb_signals)) / fps
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # --- Heart Rate vs Time ---
    if hr_times_pos:
        ax.plot(hr_times_pos, hr_values_pos, 'o-', color='#2196F3', linewidth=2, 
                markersize=4, alpha=0.85, label=f'POS (mean: {np.mean(hr_values_pos):.1f} BPM)')
    if hr_times_chrom:
        ax.plot(hr_times_chrom, hr_values_chrom, 's-', color='#FF9800', linewidth=2, 
                markersize=4, alpha=0.85, label=f'CHROM (mean: {np.mean(hr_values_chrom):.1f} BPM)')
    if hr_times_deep:
        ax.plot(hr_times_deep, hr_values_deep, 'D-', color='#4CAF50', linewidth=2,
                markersize=4, alpha=0.85, label=f'DeepPhys (mean: {np.mean(hr_values_deep):.1f} BPM)')
    
    ax.axhspan(60, 100, alpha=0.08, color='#4CAF50', label='Normal resting HR (60-100)')
    ax.axhline(y=72, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Heart Rate vs Time ({WINDOW_SEC}s window, {STEP_SEC}s step)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(35, 185)
    ax.grid(True, alpha=0.3)
    # X-axis ticks every 5 seconds
    max_t = int(time_axis[-1]) + 1
    ax.set_xticks(np.arange(0, max_t, 5))
    ax.set_xticklabels([str(t) for t in range(0, max_t, 5)], fontsize=8, rotation=45)
    
    # Watermark: proof it's from real data
    watermark = (f"Generated: {timestamp} | Video: {video_name} | "
                 f"MD5(first 64KB): {vid_hash} | FPS: {fps:.1f} | Frames: {total_frames}\n"
                 f"Reproduce: python plot_hr_vs_time.py --video {os.path.basename(video_path)}")
    fig.text(0.5, 0.01, watermark, ha='center', fontsize=7, color='gray', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#E0E0E0', alpha=0.8))
    
    plt.suptitle(f'rPPG Analysis — {video_name}\n({total_frames} frames, {duration:.1f}s @ {fps:.0f} FPS)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    out_path = f'graphs/rppg_hr_vs_time_{video_name}.png'
    plt.savefig(out_path)
    print(f"\n  Saved: {out_path}")
    plt.show()
    
    # Print summary
    if hr_values_pos:
        print(f"  POS      — Mean HR: {np.mean(hr_values_pos):.1f} BPM, "
              f"Std: {np.std(hr_values_pos):.1f}, Range: {min(hr_values_pos):.1f}-{max(hr_values_pos):.1f}")
    if hr_values_chrom:
        print(f"  CHROM    — Mean HR: {np.mean(hr_values_chrom):.1f} BPM, "
              f"Std: {np.std(hr_values_chrom):.1f}, Range: {min(hr_values_chrom):.1f}-{max(hr_values_chrom):.1f}")
    if hr_values_deep:
        print(f"  DeepPhys — Mean HR: {np.mean(hr_values_deep):.1f} BPM, "
              f"Std: {np.std(hr_values_deep):.1f}, Range: {min(hr_values_deep):.1f}-{max(hr_values_deep):.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot HR vs Time from video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--window', type=int, default=10, help='Window size in seconds (default: 10)')
    parser.add_argument('--step', type=int, default=1, help='Step size in seconds (default: 1)')
    args = parser.parse_args()
    
    WINDOW_SEC = args.window
    STEP_SEC = args.step
    
    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    os.makedirs('graphs', exist_ok=True)
    process_video(args.video)
