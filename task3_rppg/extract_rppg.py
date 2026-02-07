"""
=========================================================
 Task 3: rPPG Signal Generation
 
 Extracts heart rate signal from face videos using:
   1. POS (Plane Orthogonal to Skin) algorithm
   2. CHROM (Chrominance-based) algorithm
   3. DeepPhys (CNN with attention — Chen & McDuff, ECCV 2018)
 
 POS & CHROM are unsupervised math-based. DeepPhys is a deep learning method.
 
 Usage:
   python task3_rppg/extract_rppg.py --video 1.avi
   python task3_rppg/extract_rppg.py --video_dir data/raw/videos/Train --output_dir task3_rppg/results
   
 Output: rPPG waveform (.csv), HR estimate, signal plot (.png)
=========================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks, welch
import csv

# DeepPhys (3rd algorithm)
try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deepphys import DeepPhysExtractor
    DEEPPHYS_AVAILABLE = True
except ImportError:
    DEEPPHYS_AVAILABLE = False

# ========================= CONFIG =========================
# Bandpass filter for heart rate: 42-180 BPM → 0.7-3.0 Hz
FREQ_LOW = 0.7
FREQ_HIGH = 3.0
FILTER_ORDER = 4

# ROI: forehead + cheeks (most stable for blood flow)
# MediaPipe Face Mesh landmark indices
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132,
                93, 234, 127, 162, 21, 54, 103, 67, 109]

LEFT_CHEEK_IDX = [116, 117, 118, 119, 100, 36, 205, 187, 123, 
                  50, 101, 102, 48, 64, 235]

RIGHT_CHEEK_IDX = [345, 346, 347, 348, 329, 266, 425, 411, 352,
                   280, 330, 331, 278, 294, 455]

# ========================= FACE ROI EXTRACTION =========================
def get_roi_mean_rgb(frame, landmarks, roi_indices, img_w, img_h):
    """Extract mean RGB from a region of interest defined by landmark indices."""
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
    
    # Create mask for ROI
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Extract mean RGB
    mean_rgb = cv2.mean(frame, mask=mask)[:3]  # BGR
    return mean_rgb  # (B, G, R)

def extract_rgb_signals(video_path, collect_face_crops=False):
    """Extract per-frame mean RGB from forehead+cheeks ROI.
    If collect_face_crops=True, also returns face crops for DeepPhys (single-pass).
    """
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
        print(f"  ERROR: Cannot open {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30.0  # Cap unrealistic FPS (some videos report 1000)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    rgb_signals = []  # List of [R, G, B] per frame
    face_crops = []    # Face crops for DeepPhys (if requested)
    frame_count = 0
    faces_found = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Get mean RGB from forehead + left cheek + right cheek
            forehead_rgb = get_roi_mean_rgb(frame, lm, FOREHEAD_IDX, img_w, img_h)
            lcheek_rgb = get_roi_mean_rgb(frame, lm, LEFT_CHEEK_IDX, img_w, img_h)
            rcheek_rgb = get_roi_mean_rgb(frame, lm, RIGHT_CHEEK_IDX, img_w, img_h)
            
            # Average all ROIs (BGR format from OpenCV)
            valid_rois = [r for r in [forehead_rgb, lcheek_rgb, rcheek_rgb] if r is not None]
            if valid_rois:
                mean_bgr = np.mean(valid_rois, axis=0)
                # Convert to RGB
                rgb_signals.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])  # R, G, B
                faces_found += 1
                # Collect face crop for DeepPhys
                if collect_face_crops:
                    xs = [lm[i].x for i in range(len(lm))]
                    ys = [lm[i].y for i in range(len(lm))]
                    x1 = max(0, int(min(xs) * img_w) - 10)
                    y1 = max(0, int(min(ys) * img_h) - 10)
                    x2 = min(img_w, int(max(xs) * img_w) + 10)
                    y2 = min(img_h, int(max(ys) * img_h) + 10)
                    face_crops.append(frame[y1:y2, x1:x2].copy())
            else:
                # No valid ROI, interpolate later
                if rgb_signals:
                    rgb_signals.append(rgb_signals[-1])
                else:
                    rgb_signals.append([128, 128, 128])
                if collect_face_crops and face_crops:
                    face_crops.append(face_crops[-1])
        else:
            # No face detected - use last known value
            if rgb_signals:
                rgb_signals.append(rgb_signals[-1])
            else:
                rgb_signals.append([128, 128, 128])
            if collect_face_crops and face_crops:
                face_crops.append(face_crops[-1])
        
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"    Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    face_mesh.close()
    
    rgb_signals = np.array(rgb_signals, dtype=np.float64)  # (T, 3) = R, G, B
    print(f"  Frames: {frame_count}, Faces detected: {faces_found}/{frame_count} ({faces_found/max(1,frame_count)*100:.0f}%)")
    
    if collect_face_crops:
        return rgb_signals, fps, face_crops
    return rgb_signals, fps

# ========================= POS ALGORITHM =========================
def pos_algorithm(rgb_signal, fps, window_size=None):
    """
    POS: Plane-Orthogonal-to-Skin
    Paper: Wang et al. "Algorithmic Principles of Remote PPG" (2017)
    
    The most robust unsupervised rPPG algorithm.
    Works by projecting temporal RGB changes onto a plane orthogonal to skin tone.
    """
    if window_size is None:
        window_size = int(fps * 1.6)  # ~1.6 seconds window
    
    N = rgb_signal.shape[0]
    if N < window_size:
        window_size = N
    
    # Normalize RGB signals
    H = np.zeros(N)
    
    for t in range(window_size - 1, N):
        # Windowed segment
        C = rgb_signal[t - window_size + 1: t + 1, :]  # (W, 3)
        
        # Temporal normalization: divide by mean
        mean_C = np.mean(C, axis=0)
        mean_C[mean_C == 0] = 1  # Avoid division by zero
        Cn = C / mean_C  # (W, 3)
        
        # POS projection
        # S1 = Gn - Bn
        # S2 = Gn + Bn - 2*Rn
        S1 = Cn[:, 1] - Cn[:, 2]    # G - B
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R
        
        # Alpha
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        
        if std_S2 == 0:
            alpha = 0
        else:
            alpha = std_S1 / std_S2
        
        # Pulse signal for this window
        h = S1 + alpha * S2
        
        # Overlap-add
        H[t - window_size + 1: t + 1] += (h - np.mean(h))
    
    return H

# ========================= CHROM ALGORITHM =========================
def chrom_algorithm(rgb_signal, fps, window_size=None):
    """
    CHROM: Chrominance-based method
    Paper: De Haan & Jeanne "Robust Pulse Rate from Chrominance-Based rPPG" (2013)
    
    Uses chrominance signals (color difference) to extract pulse.
    More robust to illumination changes than Green channel.
    """
    if window_size is None:
        window_size = int(fps * 1.6)
    
    N = rgb_signal.shape[0]
    if N < window_size:
        window_size = N
    
    H = np.zeros(N)
    
    for t in range(window_size - 1, N):
        C = rgb_signal[t - window_size + 1: t + 1, :]
        
        # Temporal normalization
        mean_C = np.mean(C, axis=0)
        mean_C[mean_C == 0] = 1
        Cn = C / mean_C
        
        # CHROM projection
        # Xs = 3*Rn - 2*Gn
        # Ys = 1.5*Rn + Gn - 1.5*Bn
        Xs = 3.0 * Cn[:, 0] - 2.0 * Cn[:, 1]
        Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        
        # Alpha
        std_Xs = np.std(Xs)
        std_Ys = np.std(Ys)
        
        if std_Ys == 0:
            alpha = 0
        else:
            alpha = std_Xs / std_Ys
        
        # Pulse signal
        h = Xs - alpha * Ys
        
        H[t - window_size + 1: t + 1] += (h - np.mean(h))
    
    return H

# ========================= SIGNAL PROCESSING =========================
def bandpass_filter(signal, fps, low=FREQ_LOW, high=FREQ_HIGH, order=FILTER_ORDER):
    """Apply Butterworth bandpass filter to isolate heart rate frequencies."""
    nyquist = fps / 2.0
    
    # Clamp frequencies to valid range
    low_norm = max(low / nyquist, 0.01)
    high_norm = min(high / nyquist, 0.99)
    
    if low_norm >= high_norm:
        return signal
    
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    # Pad signal to avoid edge effects
    pad_len = min(3 * max(len(b), len(a)), len(signal) - 1)
    if pad_len < 1 or len(signal) < pad_len + 1:
        return signal
    
    filtered = filtfilt(b, a, signal, padlen=pad_len)
    return filtered

def estimate_hr(signal, fps):
    """Estimate heart rate from rPPG signal using FFT peak detection."""
    # FFT-based HR estimation
    N = len(signal)
    if N < 10:
        return 0.0
    
    # Welch's method for power spectral density
    freqs, psd = welch(signal, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
    
    # Only consider heart rate range
    valid = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if not np.any(valid):
        return 0.0
    
    freqs_valid = freqs[valid]
    psd_valid = psd[valid]
    
    # Peak frequency = heart rate
    peak_idx = np.argmax(psd_valid)
    hr_freq = freqs_valid[peak_idx]
    hr_bpm = hr_freq * 60.0
    
    return hr_bpm

def compute_signal_quality(signal, fps):
    """Compute Signal Quality Index (SNR-based)."""
    N = len(signal)
    if N < 10:
        return 0.0
    
    freqs, psd = welch(signal, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
    
    # Heart rate band power
    hr_band = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if not np.any(hr_band):
        return 0.0
    
    signal_power = np.sum(psd[hr_band])
    total_power = np.sum(psd) + 1e-10
    
    snr = signal_power / total_power
    return snr

# ========================= PLOT =========================
def plot_signals(pos_signal, chrom_signal, fps, video_name, output_dir):
    """Save plot of rPPG signals and power spectra."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend (works on server)
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return
    
    time_axis = np.arange(len(pos_signal)) / fps
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'rPPG Signal — {video_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: POS waveform
    axes[0, 0].plot(time_axis, pos_signal, 'b-', linewidth=0.5, alpha=0.8)
    axes[0, 0].set_title('POS Algorithm — rPPG Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: CHROM waveform
    axes[0, 1].plot(time_axis, chrom_signal, 'r-', linewidth=0.5, alpha=0.8)
    axes[0, 1].set_title('CHROM Algorithm — rPPG Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: POS Power Spectrum
    N = len(pos_signal)
    freqs_p, psd_p = welch(pos_signal, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
    valid_p = (freqs_p >= 0.5) & (freqs_p <= 4.0)
    axes[1, 0].plot(freqs_p[valid_p] * 60, psd_p[valid_p], 'b-')
    axes[1, 0].set_title('POS — Power Spectrum')
    axes[1, 0].set_xlabel('Heart Rate (BPM)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].axvline(x=estimate_hr(pos_signal, fps), color='g', linestyle='--', label=f'HR={estimate_hr(pos_signal, fps):.0f} BPM')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: CHROM Power Spectrum
    freqs_c, psd_c = welch(chrom_signal, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
    valid_c = (freqs_c >= 0.5) & (freqs_c <= 4.0)
    axes[1, 1].plot(freqs_c[valid_c] * 60, psd_c[valid_c], 'r-')
    axes[1, 1].set_title('CHROM — Power Spectrum')
    axes[1, 1].set_xlabel('Heart Rate (BPM)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].axvline(x=estimate_hr(chrom_signal, fps), color='g', linestyle='--', label=f'HR={estimate_hr(chrom_signal, fps):.0f} BPM')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{video_name}_rppg_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")

# ========================= MAIN PROCESSING =========================
def process_single_video(video_path, output_dir, plot=True):
    """Process one video: extract RGB → POS + CHROM + DeepPhys → filter → HR → save."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n--- Processing: {video_name} ---")
    
    # Step 1: Extract RGB signals from face ROI (+ face crops for DeepPhys)
    print("  [1/5] Extracting face ROI RGB signals...")
    if DEEPPHYS_AVAILABLE:
        rgb_signals, fps, face_crops = extract_rgb_signals(video_path, collect_face_crops=True)
    else:
        rgb_signals, fps = extract_rgb_signals(video_path, collect_face_crops=False)
        face_crops = []
    if rgb_signals is None or len(rgb_signals) < 30:
        print(f"  SKIP: Too few frames ({0 if rgb_signals is None else len(rgb_signals)})")
        return None
    
    print(f"  FPS: {fps:.1f}, Signal length: {len(rgb_signals)} frames ({len(rgb_signals)/fps:.1f}s)")
    
    # Step 2: Apply POS and CHROM algorithms
    print("  [2/5] Running POS algorithm...")
    pos_raw = pos_algorithm(rgb_signals, fps)
    
    print("  [3/5] Running CHROM algorithm...")
    chrom_raw = chrom_algorithm(rgb_signals, fps)
    
    # Step 3: Bandpass filter
    print("  [4/5] Filtering & estimating HR...")
    pos_filtered = bandpass_filter(pos_raw, fps)
    chrom_filtered = bandpass_filter(chrom_raw, fps)
    
    # Normalize signals to [-1, 1]
    pos_max = np.max(np.abs(pos_filtered)) + 1e-10
    chrom_max = np.max(np.abs(chrom_filtered)) + 1e-10
    pos_norm = pos_filtered / pos_max
    chrom_norm = chrom_filtered / chrom_max
    
    # Step 4: Estimate HR
    hr_pos = estimate_hr(pos_filtered, fps)
    hr_chrom = estimate_hr(chrom_filtered, fps)
    
    # Signal quality
    sqi_pos = compute_signal_quality(pos_filtered, fps)
    sqi_chrom = compute_signal_quality(chrom_filtered, fps)
    
    # Step 5: DeepPhys from pre-extracted face crops (no video re-reading!)
    hr_deep, sqi_deep = 0.0, 0.0
    if DEEPPHYS_AVAILABLE and len(face_crops) > int(fps * 5):
        try:
            print("  [5/5] Running DeepPhys (from cached face crops)...")
            extractor = DeepPhysExtractor()
            hr_deep, sqi_deep, _ = extractor.extract_hr_from_frames(face_crops, fps)
        except Exception as e:
            print(f"  DeepPhys error: {e}")
    
    print(f"  POS     → HR: {hr_pos:.1f} BPM | SQI: {sqi_pos:.3f}")
    print(f"  CHROM   → HR: {hr_chrom:.1f} BPM | SQI: {sqi_chrom:.3f}")
    print(f"  DeepPhys→ HR: {hr_deep:.1f} BPM | SQI: {sqi_deep:.3f}")
    
    # Plausibility check
    for algo, hr in [("POS", hr_pos), ("CHROM", hr_chrom), ("DeepPhys", hr_deep)]:
        if 50 <= hr <= 120:
            print(f"  {algo}: Plausible HR ✓")
        else:
            print(f"  {algo}: Unusual HR (outside 50-120 BPM range)")
    
    # Save signals — compact .npy format (much smaller than CSV)
    os.makedirs(output_dir, exist_ok=True)
    
    # Downsample to max 128 points to save space
    def downsample(sig, target=128):
        if len(sig) <= target:
            return sig
        indices = np.linspace(0, len(sig)-1, target, dtype=int)
        return sig[indices]
    
    pos_ds = downsample(pos_norm)
    chrom_ds = downsample(chrom_norm)
    
    # Save as single compact .npy (both signals + metadata)
    save_data = {
        'pos_signal': pos_ds.astype(np.float32),
        'chrom_signal': chrom_ds.astype(np.float32),
        'hr_pos': hr_pos,
        'hr_chrom': hr_chrom,
        'hr_deepphys': hr_deep,
        'sqi_pos': sqi_pos,
        'sqi_chrom': sqi_chrom,
        'sqi_deepphys': sqi_deep,
        'fps': fps,
        'frames': len(rgb_signals)
    }
    npy_path = os.path.join(output_dir, f'{video_name}_rppg.npy')
    np.save(npy_path, save_data)
    print(f"  Saved: {npy_path}")
    
    # Plot (skip to save space)
    if plot:
        plot_signals(pos_norm, chrom_norm, fps, video_name, output_dir)
    
    return {
        'video': video_name,
        'fps': fps,
        'frames': len(rgb_signals),
        'hr_pos': hr_pos,
        'hr_chrom': hr_chrom,
        'hr_deepphys': hr_deep,
        'sqi_pos': sqi_pos,
        'sqi_chrom': sqi_chrom,
        'sqi_deepphys': sqi_deep
    }

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="rPPG Signal Extraction (POS + CHROM + DeepPhys)")
    parser.add_argument('--video', type=str, default=None, help="Path to single video")
    parser.add_argument('--video_dir', type=str, default=None, help="Path to directory of videos")
    parser.add_argument('--output_dir', type=str, default='task3_rppg/results', help="Output directory")
    parser.add_argument('--no_plot', action='store_true', help="Skip saving plots")
    args = parser.parse_args()
    
    if args.video is None and args.video_dir is None:
        print("ERROR: Provide --video or --video_dir")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect video paths
    videos = []
    if args.video:
        videos = [args.video]
    else:
        exts = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm')
        for f in sorted(os.listdir(args.video_dir)):
            if f.lower().endswith(exts):
                videos.append(os.path.join(args.video_dir, f))
        print(f"Found {len(videos)} videos in {args.video_dir}")
    
    # Step 1: Convert existing CSVs to .npy (recover previous work)
    recovered = 0
    for vpath in videos:
        video_name = os.path.splitext(os.path.basename(vpath))[0]
        npy_path = os.path.join(args.output_dir, f'{video_name}_rppg.npy')
        pos_csv = os.path.join(args.output_dir, f'{video_name}_pos.csv')
        chrom_csv = os.path.join(args.output_dir, f'{video_name}_chrom.csv')
        
        # If .npy already exists, skip
        if os.path.exists(npy_path):
            continue
        
        # If CSVs exist, convert to .npy and delete CSVs
        if os.path.exists(pos_csv) and os.path.exists(chrom_csv):
            try:
                import pandas as pd
                pos_df = pd.read_csv(pos_csv)
                chrom_df = pd.read_csv(chrom_csv)
                pos_sig = pos_df['rppg_signal'].values.astype(np.float32)
                chrom_sig = chrom_df['rppg_signal'].values.astype(np.float32)
                fps_est = 30.0  # Default, CSVs don't store fps
                if len(pos_df) > 1:
                    dt = pos_df['time_s'].iloc[1] - pos_df['time_s'].iloc[0]
                    if dt > 0:
                        fps_est = 1.0 / dt
                
                # Downsample
                def ds(sig, target=128):
                    if len(sig) <= target: return sig
                    idx = np.linspace(0, len(sig)-1, target, dtype=int)
                    return sig[idx]
                
                # Compute HR/SQI from recovered signal
                hr_p = estimate_hr(pos_sig, fps_est)
                hr_c = estimate_hr(chrom_sig, fps_est)
                sqi_p = compute_signal_quality(pos_sig, fps_est)
                sqi_c = compute_signal_quality(chrom_sig, fps_est)
                
                save_data = {
                    'pos_signal': ds(pos_sig),
                    'chrom_signal': ds(chrom_sig),
                    'hr_pos': hr_p, 'hr_chrom': hr_c,
                    'sqi_pos': sqi_p, 'sqi_chrom': sqi_c,
                    'fps': fps_est, 'frames': len(pos_sig)
                }
                np.save(npy_path, save_data)
                
                # Delete CSVs to free space
                os.remove(pos_csv)
                os.remove(chrom_csv)
                recovered += 1
            except Exception as e:
                print(f"  Warning: Could not recover {video_name}: {e}")
    
    if recovered > 0:
        print(f"Recovered {recovered} videos from existing CSVs → .npy (CSVs deleted to free space)")
    
    # Step 2: Process remaining videos (skip already done)
    all_results = []
    skipped = 0
    for vpath in videos:
        video_name = os.path.splitext(os.path.basename(vpath))[0]
        npy_path = os.path.join(args.output_dir, f'{video_name}_rppg.npy')
        
        # Skip if already processed
        if os.path.exists(npy_path):
            try:
                data = np.load(npy_path, allow_pickle=True).item()
                all_results.append({
                    'video': video_name, 'fps': data['fps'], 'frames': data['frames'],
                    'hr_pos': data['hr_pos'], 'hr_chrom': data['hr_chrom'],
                    'hr_deepphys': data.get('hr_deepphys', 0.0),
                    'sqi_pos': data['sqi_pos'], 'sqi_chrom': data['sqi_chrom'],
                    'sqi_deepphys': data.get('sqi_deepphys', 0.0)
                })
                skipped += 1
            except:
                pass
            continue
        
        if not os.path.exists(vpath):
            print(f"SKIP: {vpath} not found")
            continue
        result = process_single_video(vpath, args.output_dir, plot=not args.no_plot)
        if result:
            all_results.append(result)
    
    if skipped > 0:
        print(f"\nSkipped {skipped} already processed videos, processed {len(all_results)-skipped} new")
    
    # Summary
    if all_results:
        print("\n" + "=" * 65)
        print(f"{'SUMMARY':^65}")
        print("=" * 65)
        print(f"{'Video':<15} {'HR_POS':>8} {'HR_CHROM':>10} {'HR_DEEP':>9} {'SQI_POS':>9} {'SQI_CHROM':>11} {'SQI_DEEP':>10}")
        print("-" * 85)
        
        plausible_pos = 0
        plausible_chrom = 0
        plausible_deep = 0
        
        for r in all_results:
            pos_ok = "✓" if 50 <= r['hr_pos'] <= 120 else "✗"
            chrom_ok = "✓" if 50 <= r['hr_chrom'] <= 120 else "✗"
            deep_ok = "✓" if 50 <= r.get('hr_deepphys', 0) <= 120 else "✗"
            print(f"{r['video']:<15} {r['hr_pos']:>6.1f}{pos_ok:>2} {r['hr_chrom']:>8.1f}{chrom_ok:>2} {r.get('hr_deepphys',0):>7.1f}{deep_ok:>2} {r['sqi_pos']:>9.3f} {r['sqi_chrom']:>11.3f} {r.get('sqi_deepphys',0):>10.3f}")
            if 50 <= r['hr_pos'] <= 120: plausible_pos += 1
            if 50 <= r['hr_chrom'] <= 120: plausible_chrom += 1
            if 50 <= r.get('hr_deepphys', 0) <= 120: plausible_deep += 1
        
        print("-" * 85)
        total = len(all_results)
        print(f"POS      Plausible HR: {plausible_pos}/{total} ({plausible_pos/total*100:.0f}%)")
        print(f"CHROM    Plausible HR: {plausible_chrom}/{total} ({plausible_chrom/total*100:.0f}%)")
        print(f"DeepPhys Plausible HR: {plausible_deep}/{total} ({plausible_deep/total*100:.0f}%)")
        
        avg_sqi_pos = np.mean([r['sqi_pos'] for r in all_results])
        avg_sqi_chrom = np.mean([r['sqi_chrom'] for r in all_results])
        avg_sqi_deep = np.mean([r.get('sqi_deepphys', 0) for r in all_results])
        print(f"POS      Avg SQI: {avg_sqi_pos:.3f}")
        print(f"CHROM    Avg SQI: {avg_sqi_chrom:.3f}")
        print(f"DeepPhys Avg SQI: {avg_sqi_deep:.3f}")
        
        # Best algorithm
        sqi_map = {'POS': avg_sqi_pos, 'CHROM': avg_sqi_chrom, 'DeepPhys': avg_sqi_deep}
        best = max(sqi_map, key=sqi_map.get)
        print(f"\nBest Algorithm: {best}")
        print("=" * 85)
        
        # Save summary CSV
        summary_path = os.path.join(args.output_dir, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'fps', 'frames', 'hr_pos', 'hr_chrom', 'hr_deepphys', 'sqi_pos', 'sqi_chrom', 'sqi_deepphys'])
            for r in all_results:
                writer.writerow([r['video'], r['fps'], r['frames'], 
                                round(r['hr_pos'], 2), round(r['hr_chrom'], 2), round(r.get('hr_deepphys', 0), 2),
                                round(r['sqi_pos'], 4), round(r['sqi_chrom'], 4), round(r.get('sqi_deepphys', 0), 4)])
        print(f"Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
