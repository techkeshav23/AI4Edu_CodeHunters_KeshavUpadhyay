"""
=========================================================
 rPPG Feature Extraction for Multimodal Fusion (Task 4)
 
 Extracts physiological features from existing rPPG results:
   - HR Proxy (3 algos)
   - HRV approximation (from BVP peaks)
   - SQI (Signal Quality Index)
   - Bandpower LF/HF ratio
   - HR consensus & variability
 
 Input:  task3_rppg/results/*.npy (from Task 3)
 Output: data/rppg_features/*.npy (per-video feature vector)
 
 Usage:
   python task4_5_multimodal/extract_rppg_features.py \
     --results_dir task3_rppg/results \
     --output_dir data/rppg_features
=========================================================
"""

import os
import argparse
import numpy as np
from scipy.signal import find_peaks, welch

# rPPG feature vector = 14 dimensions
RPPG_FEATURE_DIM = 14


def extract_hrv_features(bvp_signal, fps):
    """
    Extract HRV-like features from BVP signal.
    
    Even with 128-point downsampled signal, we can get basic HRV:
    - SDNN: Std of peak-to-peak intervals
    - RMSSD: Root mean square of successive differences
    - Mean IBI: Mean inter-beat interval
    """
    if bvp_signal is None or len(bvp_signal) < 10:
        return 0.0, 0.0, 0.0
    
    # Find peaks in BVP signal
    # Distance: at least 0.4 sec apart (150 BPM max)
    min_dist = max(1, int(0.4 * fps))
    peaks, _ = find_peaks(bvp_signal, distance=min_dist, height=0)
    
    if len(peaks) < 3:
        return 0.0, 0.0, 0.0
    
    # Inter-beat intervals (in seconds)
    ibis = np.diff(peaks) / fps
    
    # Filter physiologically plausible IBIs (0.33s to 1.5s = 40-180 BPM)
    valid_ibis = ibis[(ibis > 0.33) & (ibis < 1.5)]
    
    if len(valid_ibis) < 2:
        return 0.0, 0.0, 0.0
    
    mean_ibi = float(np.mean(valid_ibis))
    sdnn = float(np.std(valid_ibis))
    
    # RMSSD
    successive_diffs = np.diff(valid_ibis)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
    
    return mean_ibi, sdnn, rmssd


def extract_bandpower(bvp_signal, fps):
    """
    Extract LF/HF ratio from BVP signal.
    
    LF (0.04-0.15 Hz): Sympathetic + parasympathetic
    HF (0.15-0.4 Hz): Parasympathetic (relaxation)
    LF/HF ratio: Higher = more stress/cognitive load
    """
    if bvp_signal is None or len(bvp_signal) < 10:
        return 0.0, 0.0, 0.0
    
    try:
        nperseg = min(64, len(bvp_signal))
        noverlap = nperseg // 2
        freqs, psd = welch(bvp_signal, fs=fps, nperseg=nperseg,
                           noverlap=noverlap)
        
        # LF band: 0.04-0.15 Hz
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
        
        # HF band: 0.15-0.4 Hz
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
        
        # LF/HF ratio
        lf_hf_ratio = lf_power / (hf_power + 1e-10)
        
        return lf_power, hf_power, lf_hf_ratio
    except Exception:
        return 0.0, 0.0, 0.0


def extract_rppg_feature_vector(data):
    """
    Extract 14-dim rPPG feature vector from one .npy result dict.
    
    Feature vector:
      [0]  hr_pos          - HR from POS algorithm
      [1]  hr_chrom        - HR from CHROM algorithm
      [2]  hr_deepphys     - HR from DeepPhys CNN
      [3]  sqi_pos         - Signal quality POS
      [4]  sqi_chrom       - Signal quality CHROM
      [5]  sqi_deepphys    - Signal quality DeepPhys
      [6]  hr_best         - Best HR (highest SQI)
      [7]  hr_std          - Std across 3 HR estimates (consensus)
      [8]  mean_ibi        - Mean inter-beat interval (HRV)
      [9]  sdnn            - SDNN (HRV)
      [10] rmssd           - RMSSD (HRV)
      [11] lf_power        - Low frequency power
      [12] hf_power        - High frequency power
      [13] lf_hf_ratio     - LF/HF ratio (stress indicator)
    """
    # HR and SQI from 3 algorithms
    hr_pos = data.get('hr_pos', 0.0)
    hr_chrom = data.get('hr_chrom', 0.0)
    hr_deep = data.get('hr_deepphys', 0.0)
    sqi_pos = data.get('sqi_pos', 0.0)
    sqi_chrom = data.get('sqi_chrom', 0.0)
    sqi_deep = data.get('sqi_deepphys', 0.0)
    
    # Best HR (weighted by SQI)
    hrs = [hr_pos, hr_chrom, hr_deep]
    sqis = [sqi_pos, sqi_chrom, sqi_deep]
    hr_best = hrs[int(np.argmax(sqis))] if max(sqis) > 0 else np.mean(hrs)
    
    # HR consensus (std — low = algorithms agree)
    valid_hrs = [h for h in hrs if h > 0]
    hr_std = float(np.std(valid_hrs)) if len(valid_hrs) > 1 else 0.0
    
    # HRV features from best BVP signal
    fps = data.get('fps', 30.0)
    # Effective fps for downsampled 128-point signal
    frames = data.get('frames', 1)
    effective_fps = 128.0 * fps / max(frames, 1) if frames > 128 else fps
    
    # Use POS signal (usually most reliable)
    bvp = data.get('pos_signal', None)
    if bvp is None:
        bvp = data.get('chrom_signal', None)
    
    mean_ibi, sdnn, rmssd = extract_hrv_features(bvp, effective_fps)
    lf_power, hf_power, lf_hf_ratio = extract_bandpower(bvp, effective_fps)
    
    feature = np.array([
        hr_pos, hr_chrom, hr_deep,
        sqi_pos, sqi_chrom, sqi_deep,
        hr_best, hr_std,
        mean_ibi, sdnn, rmssd,
        lf_power, hf_power, lf_hf_ratio
    ], dtype=np.float32)
    
    return feature


def main():
    parser = argparse.ArgumentParser(description="Extract rPPG features for multimodal fusion")
    parser.add_argument('--results_dir', type=str, default='task3_rppg/results',
                        help='Directory with rPPG .npy result files')
    parser.add_argument('--output_dir', type=str, default='data/rppg_features',
                        help='Output directory for feature .npy files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    npy_files = sorted([f for f in os.listdir(args.results_dir) if f.endswith('_rppg.npy')])
    print(f"Found {len(npy_files)} rPPG result files")
    print(f"Output feature dim: {RPPG_FEATURE_DIM}")
    
    for i, npy_name in enumerate(npy_files):
        video_name = npy_name.replace('_rppg.npy', '')
        npy_path = os.path.join(args.results_dir, npy_name)
        
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            feature = extract_rppg_feature_vector(data)
            
            out_path = os.path.join(args.output_dir, video_name + '.npy')
            np.save(out_path, feature)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(npy_files)}] {video_name}: "
                      f"HR={feature[6]:.1f} HRV_SDNN={feature[9]:.4f} LF/HF={feature[13]:.2f}")
        except Exception as e:
            print(f"  [{i+1}/{len(npy_files)}] {video_name}: ERROR — {e}")
    
    print(f"\nDone! Features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
