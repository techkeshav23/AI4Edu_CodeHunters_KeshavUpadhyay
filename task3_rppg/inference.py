"""
=========================================================
 Task 3: rPPG Inference & Evaluation
 
 Extracts rPPG heart rate from test videos using POS + CHROM.
 Computes MAE (BPM) and Pearson Correlation if ground truth given.
 
 Usage:
   # Inference on test videos (no ground truth):
   python task3_rppg/inference.py --test_dir dataset/test/videos
   
   # Evaluation with ground truth:
   python task3_rppg/inference.py --test_dir dataset/test/videos --ground_truth dataset/test/labels_test.xlsx
   
 Output:
   - rppg_predictions.csv (video, hr_pos, hr_chrom, hr_deepphys, sqi_pos, sqi_chrom, sqi_deepphys)
   - MAE + Pearson printed if ground truth provided
=========================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr
import csv
import warnings
warnings.filterwarnings("ignore")

# DeepPhys (3rd algorithm)
try:
    from deepphys import DeepPhysExtractor
    DEEPPHYS_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from deepphys import DeepPhysExtractor
        DEEPPHYS_AVAILABLE = True
    except ImportError:
        DEEPPHYS_AVAILABLE = False
        print("Warning: DeepPhys not available. Using POS + CHROM only.")

# ========================= CONFIG =========================
FREQ_LOW = 0.7
FREQ_HIGH = 3.0
FILTER_ORDER = 4
VIDEO_EXTS = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm')

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

# ========================= rPPG FUNCTIONS =========================
def get_roi_mean_rgb(frame, landmarks, roi_indices, img_w, img_h):
    points = []
    for idx in roi_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            points.append([int(lm.x * img_w), int(lm.y * img_h)])
    if len(points) < 3:
        return None
    points = np.array(points, dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.mean(frame, mask=mask)[:3]  # BGR

def bandpass_filter(signal, fs):
    nyq = 0.5 * fs
    low_n, high_n = FREQ_LOW / nyq, FREQ_HIGH / nyq
    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        return signal
    b, a = butter(FILTER_ORDER, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))

def pos_algorithm(rgb):
    if len(rgb) < 10: return np.zeros(len(rgb))
    mean_rgb = np.mean(rgb, axis=0, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1
    Cn = rgb / mean_rgb
    S1 = Cn[:, 1] - Cn[:, 2]
    S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
    std2 = np.std(S2)
    if std2 == 0: return S1
    return S1 + (np.std(S1) / std2) * S2

def chrom_algorithm(rgb):
    if len(rgb) < 10: return np.zeros(len(rgb))
    mean_rgb = np.mean(rgb, axis=0, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1
    Cn = rgb / mean_rgb
    Xs = 3 * Cn[:, 0] - 2 * Cn[:, 1]
    Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
    std_y = np.std(Ys)
    if std_y == 0: return Xs
    return Xs - (np.std(Xs) / std_y) * Ys

def estimate_hr(signal, fs):
    """Estimate heart rate from rPPG signal using FFT."""
    if len(signal) < int(fs * 3):
        return 0.0, 0.0
    try:
        filtered = bandpass_filter(signal, fs)
        freqs, psd = welch(filtered, fs=fs, nperseg=min(512, len(filtered)),
                          noverlap=min(256, len(filtered)//2))
        valid = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
        if not np.any(valid) or np.max(psd[valid]) == 0:
            return 0.0, 0.0
        peak_freq = freqs[valid][np.argmax(psd[valid])]
        hr = peak_freq * 60
        # Signal Quality Index
        peak_power = np.max(psd[valid])
        total_power = np.sum(psd[valid])
        sqi = peak_power / total_power if total_power > 0 else 0
        autocorr = np.correlate(filtered, filtered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        sqi = max(sqi, np.max(autocorr[1:]) if len(autocorr) > 1 else 0)
        return hr, min(sqi, 1.0)
    except:
        return 0.0, 0.0

# Global DeepPhys extractor (initialized once)
_deepphys_extractor = None

def get_deepphys_extractor(weights_path=None):
    global _deepphys_extractor
    if _deepphys_extractor is None and DEEPPHYS_AVAILABLE:
        _deepphys_extractor = DeepPhysExtractor(weights_path=weights_path)
    return _deepphys_extractor

def process_single_video(video_path, run_deepphys=True):
    """Extract rPPG HR from a single video using POS + CHROM + DeepPhys (single pass)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Cannot open {video_path}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    rgb_signals = []
    face_crops = []  # For DeepPhys — collected in same pass
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            mean_bgr = get_roi_mean_rgb(frame, lm, ALL_ROI_IDX, img_w, img_h)
            if mean_bgr is not None:
                rgb_signals.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
            elif rgb_signals:
                rgb_signals.append(rgb_signals[-1])
            
            # Extract face bounding box from landmarks for DeepPhys
            if run_deepphys and DEEPPHYS_AVAILABLE:
                xs = [lm[i].x for i in range(len(lm))]
                ys = [lm[i].y for i in range(len(lm))]
                x1 = max(0, int(min(xs) * img_w) - 10)
                y1 = max(0, int(min(ys) * img_h) - 10)
                x2 = min(img_w, int(max(xs) * img_w) + 10)
                y2 = min(img_h, int(max(ys) * img_h) + 10)
                face_crops.append(frame[y1:y2, x1:x2].copy())
        else:
            if rgb_signals:
                rgb_signals.append(rgb_signals[-1])
            if run_deepphys and face_crops:
                face_crops.append(face_crops[-1])
        
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"    Frame {frame_count}/{total_frames}")
    
    cap.release()
    face_mesh.close()
    
    if len(rgb_signals) < int(fps * 5):
        print(f"    Too few frames with face: {len(rgb_signals)}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    rgb_signals = np.array(rgb_signals)
    pos_sig = pos_algorithm(rgb_signals)
    chrom_sig = chrom_algorithm(rgb_signals)
    hr_pos, sqi_pos = estimate_hr(pos_sig, fps)
    hr_chrom, sqi_chrom = estimate_hr(chrom_sig, fps)
    
    # DeepPhys from pre-extracted face crops (no video re-reading!)
    hr_deep, sqi_deep = 0.0, 0.0
    if run_deepphys and DEEPPHYS_AVAILABLE and len(face_crops) > int(fps * 5):
        try:
            extractor = get_deepphys_extractor()
            hr_deep, sqi_deep, _ = extractor.extract_hr_from_frames(face_crops, fps)
            print(f"    DeepPhys: {hr_deep:.1f} BPM (SQI: {sqi_deep:.4f})")
        except Exception as e:
            print(f"    DeepPhys error: {e}")
    
    return hr_pos, hr_chrom, sqi_pos, sqi_chrom, hr_deep, sqi_deep

# ========================= EVALUATION =========================
def compute_metrics(pred_hr, gt_hr):
    """Compute MAE (BPM) and Pearson Correlation."""
    pred_hr = np.array(pred_hr, dtype=float)
    gt_hr = np.array(gt_hr, dtype=float)
    valid = (pred_hr > 0) & (gt_hr > 0)
    pred_valid = pred_hr[valid]
    gt_valid = gt_hr[valid]
    if len(pred_valid) < 2:
        return float('inf'), 0.0, 0
    mae = np.mean(np.abs(pred_valid - gt_valid))
    r, _ = pearsonr(pred_valid, gt_valid)
    return mae, r, len(pred_valid)

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Task 3: rPPG Inference & Evaluation")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test videos")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Excel/CSV with ground truth HR (columns: video, hr)")
    parser.add_argument("--output", type=str, default="rppg_predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--algorithm", type=str, default="best",
                        choices=["pos", "chrom", "deepphys", "best"],
                        help="Which algorithm for final HR (default: best)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max videos to process (0 = all)")
    parser.add_argument("--deepphys_weights", type=str, default=None,
                        help="Path to DeepPhys pretrained weights")
    parser.add_argument("--no_deepphys", action="store_true",
                        help="Skip DeepPhys (faster, POS+CHROM only)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Task 3: rPPG Heart Rate Extraction")
    print(" Algorithms: POS + CHROM + DeepPhys")
    print("=" * 60)
    
    # Initialize DeepPhys
    run_deepphys = DEEPPHYS_AVAILABLE and not args.no_deepphys
    if run_deepphys:
        get_deepphys_extractor(args.deepphys_weights)
        print("DeepPhys: ENABLED")
    else:
        print("DeepPhys: DISABLED" + (" (not available)" if not DEEPPHYS_AVAILABLE else " (--no_deepphys)"))
    
    # Collect test videos
    videos = []
    for f in sorted(os.listdir(args.test_dir)):
        if f.lower().endswith(VIDEO_EXTS):
            videos.append(os.path.join(args.test_dir, f))
    
    if args.limit > 0:
        videos = videos[:args.limit]
    print(f"Processing {len(videos)} videos from {args.test_dir}")
    if len(videos) == 0:
        print("ERROR: No videos found!")
        sys.exit(1)
    
    # Process each video
    results = []
    for i, vpath in enumerate(videos):
        vname = os.path.basename(vpath)
        print(f"\n[{i+1}/{len(videos)}] {vname}")
        hr_pos, hr_chrom, sqi_pos, sqi_chrom, hr_deep, sqi_deep = process_single_video(vpath, run_deepphys)
        
        # Select best HR based on algorithm choice
        if args.algorithm == "pos":
            hr_final = hr_pos
        elif args.algorithm == "chrom":
            hr_final = hr_chrom
        elif args.algorithm == "deepphys":
            hr_final = hr_deep if hr_deep > 0 else hr_chrom
        else:  # "best" — pick highest SQI among all 3
            candidates = [(hr_pos, sqi_pos), (hr_chrom, sqi_chrom), (hr_deep, sqi_deep)]
            candidates = [(hr, sqi) for hr, sqi in candidates if hr > 0]
            if candidates:
                hr_final = max(candidates, key=lambda x: x[1])[0]
            else:
                hr_final = 0.0
        
        results.append({
            'video': os.path.splitext(vname)[0],
            'hr_predicted': round(hr_final, 2),
            'hr_pos': round(hr_pos, 2),
            'hr_chrom': round(hr_chrom, 2),
            'hr_deepphys': round(hr_deep, 2),
            'sqi_pos': round(sqi_pos, 4),
            'sqi_chrom': round(sqi_chrom, 4),
            'sqi_deepphys': round(sqi_deep, 4),
        })
        print(f"    POS: {hr_pos:.1f} | CHROM: {hr_chrom:.1f} | DeepPhys: {hr_deep:.1f} | Final: {hr_final:.1f} BPM")
    
    # Save predictions
    fieldnames = ['video', 'hr_predicted', 'hr_pos', 'hr_chrom', 'hr_deepphys', 'sqi_pos', 'sqi_chrom', 'sqi_deepphys']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPredictions saved to {args.output}")
    
    # Evaluate if ground truth provided
    if args.ground_truth:
        print(f"\n{'=' * 60}")
        print(" EVALUATION: MAE (BPM) + Pearson Correlation")
        print(f"{'=' * 60}")
        import pandas as pd
        gt_df = pd.read_excel(args.ground_truth) if args.ground_truth.endswith('.xlsx') else pd.read_csv(args.ground_truth)
        gt_dict = {os.path.splitext(str(r['video']))[0]: float(r['hr']) for _, r in gt_df.iterrows()}
        
        pred_hr = [r['hr_predicted'] for r in results if r['video'] in gt_dict]
        gt_hr = [gt_dict[r['video']] for r in results if r['video'] in gt_dict]
        
        if pred_hr:
            mae, pearson_r, n_valid = compute_metrics(pred_hr, gt_hr)
            print(f"\n  Videos matched: {len(pred_hr)} | Valid: {n_valid}")
            print(f"  ┌─────────────────────────────────┐")
            print(f"  │  MAE (BPM):           {mae:>8.2f}  │")
            print(f"  │  Pearson Correlation:  {pearson_r:>8.4f}  │")
            print(f"  └─────────────────────────────────┘")
            
            mae_p, r_p, _ = compute_metrics(
                [r['hr_pos'] for r in results if r['video'] in gt_dict], gt_hr)
            mae_c, r_c, _ = compute_metrics(
                [r['hr_chrom'] for r in results if r['video'] in gt_dict], gt_hr)
            mae_d, r_d, _ = compute_metrics(
                [r['hr_deepphys'] for r in results if r['video'] in gt_dict], gt_hr)
            print(f"\n  POS      → MAE: {mae_p:.2f} BPM, Pearson: {r_p:.4f}")
            print(f"  CHROM    → MAE: {mae_c:.2f} BPM, Pearson: {r_c:.4f}")
            print(f"  DeepPhys → MAE: {mae_d:.2f} BPM, Pearson: {r_d:.4f}")
    
    # Summary
    hr_list = [r['hr_predicted'] for r in results if r['hr_predicted'] > 0]
    plausible = [h for h in hr_list if 40 <= h <= 120]
    print(f"\n{'=' * 60}")
    print(f" SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total: {len(results)} | HR extracted: {len(hr_list)} | "
          f"Plausible: {len(plausible)} ({len(plausible)/max(len(results),1)*100:.1f}%)")
    if hr_list:
        print(f"  Mean HR: {np.mean(hr_list):.1f} BPM | Median: {np.median(hr_list):.1f} BPM")

if __name__ == "__main__":
    main()
