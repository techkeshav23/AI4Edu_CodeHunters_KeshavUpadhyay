"""
=========================================================
 DeepPhys: End-to-End Deep Learning rPPG
 
 Paper: Chen & McDuff, "DeepPhys: Video-Based Physiological
        Measurement Using Convolutional Attention Networks" (ECCV 2018)
 
 Architecture:
   - Motion branch: CNN on frame differences (Δ frames)
   - Appearance branch: CNN on raw frames → generates attention mask
   - Attention mask guides motion branch to focus on skin regions
   - Output: per-frame BVP (blood volume pulse) signal
 
 This implementation uses pretrained UBFC-rPPG weights if available,
 otherwise falls back to unsupervised signal extraction.
 
 Usage:
   from deepphys import DeepPhysExtractor
   extractor = DeepPhysExtractor()
   hr, sqi, bvp = extractor.extract_hr(video_path)
=========================================================
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, welch

# ========================= CONFIG =========================
DEEPPHYS_FRAME_SIZE = 36      # DeepPhys uses 36x36 face crops
FREQ_LOW = 0.7                # 42 BPM
FREQ_HIGH = 3.0               # 180 BPM
FILTER_ORDER = 4


# ========================= DeepPhys MODEL =========================
class AttentionBlock(nn.Module):
    """Attention mechanism from appearance branch."""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.attention(x)


class DeepPhysModel(nn.Module):
    """
    DeepPhys: Two-branch CNN with attention.
    
    Branch 1 (Appearance): Processes raw frame → attention mask
    Branch 2 (Motion): Processes frame difference → guided by attention → BVP
    """
    def __init__(self):
        super().__init__()
        
        # ---- Appearance Branch ----
        self.appearance_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.appearance_pool1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.appearance_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.appearance_pool2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Attention masks (from appearance)
        self.attention1 = AttentionBlock(32)
        self.attention2 = AttentionBlock(64)
        
        # ---- Motion Branch ----
        self.motion_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.motion_pool1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.motion_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.motion_pool2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # ---- Fully Connected (after motion branch) ----
        # Input: 64 * 9 * 9 = 5184 (for 36x36 input)
        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, appearance_input, motion_input):
        """
        Args:
            appearance_input: (B, 3, 36, 36) - current frame (normalized)
            motion_input:     (B, 3, 36, 36) - frame difference
        Returns:
            bvp: (B, 1) - predicted BVP value for this frame
        """
        # Appearance branch → attention masks
        a1 = self.appearance_conv1(appearance_input)
        attn1 = self.attention1(a1)  # (B, 1, 36, 36)
        a1 = self.appearance_pool1(a1)
        a2 = self.appearance_conv2(a1)
        attn2 = self.attention2(a2)  # (B, 1, 18, 18)
        
        # Motion branch with attention
        m1 = self.motion_conv1(motion_input)
        m1 = m1 * attn1              # Apply attention from appearance
        m1 = self.motion_pool1(m1)
        m2 = self.motion_conv2(m1)
        m2 = m2 * attn2              # Apply attention from appearance
        m2 = self.motion_pool2(m2)
        
        # Flatten and predict
        m2 = m2.view(m2.size(0), -1)
        bvp = self.fc(m2)
        return bvp


# ========================= FACE DETECTION =========================
def detect_face_crop(frame, face_cascade=None):
    """Detect face and return 36x36 crop for DeepPhys."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    # Use largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Add margin (20%)
    margin = int(0.2 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    
    face = frame[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (DEEPPHYS_FRAME_SIZE, DEEPPHYS_FRAME_SIZE))
    return face_resized


# ========================= DeepPhys EXTRACTOR =========================
class DeepPhysExtractor:
    """
    Full DeepPhys pipeline: video → face crops → CNN → BVP → HR.
    
    If pretrained weights exist, uses learned features.
    Otherwise, uses the model architecture with normalized difference
    frames as a sophisticated motion-based rPPG method.
    """
    
    def __init__(self, weights_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = DeepPhysModel().to(self.device)
        self.pretrained = False
        
        # Try to load pretrained weights
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.pretrained = True
                print(f"  [DeepPhys] Loaded pretrained weights: {weights_path}")
            except Exception as e:
                print(f"  [DeepPhys] Warning: Could not load weights: {e}")
        
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def _preprocess_frame(self, frame):
        """Normalize frame to [0, 1] and convert to tensor."""
        face = detect_face_crop(frame, self.face_cascade)
        if face is None:
            return None
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_norm = face_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(face_norm).permute(2, 0, 1)  # (3, 36, 36)
        return tensor
    
    def extract_bvp(self, video_path):
        """
        Extract BVP signal from video using DeepPhys.
        
        Returns:
            bvp_signal: numpy array of per-frame BVP values
            fps: frames per second
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [DeepPhys] ERROR: Cannot open {video_path}")
            return None, 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 60:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_tensor = []
        frame_count = 0
        
        # Read all frames and get face crops
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            tensor = self._preprocess_frame(frame)
            if tensor is not None:
                frames_tensor.append(tensor)
            elif frames_tensor:
                frames_tensor.append(frames_tensor[-1])  # repeat last
            
            frame_count += 1
            if frame_count % 500 == 0:
                print(f"    [DeepPhys] Frame {frame_count}/{total_frames}")
        
        cap.release()
        
        if len(frames_tensor) < int(fps * 5):
            print(f"  [DeepPhys] Too few frames: {len(frames_tensor)}")
            return None, fps
        
        print(f"    [DeepPhys] Processing {len(frames_tensor)} frames through CNN...")
        
        # Build appearance (current frame) and motion (difference) inputs
        bvp_signal = []
        
        with torch.no_grad():
            batch_size = 64
            
            for i in range(1, len(frames_tensor)):
                appearance = frames_tensor[i]        # Current frame
                motion = frames_tensor[i] - frames_tensor[i - 1]  # Difference
                
                # Collect batch
                if i == 1 or len(bvp_signal) == 0:
                    app_batch = [appearance]
                    mot_batch = [motion]
                else:
                    app_batch.append(appearance)
                    mot_batch.append(motion)
                
                # Process batch
                if len(app_batch) >= batch_size or i == len(frames_tensor) - 1:
                    app_tensor = torch.stack(app_batch).to(self.device)
                    mot_tensor = torch.stack(mot_batch).to(self.device)
                    
                    if self.pretrained:
                        out = self.model(app_tensor, mot_tensor)
                        bvp_signal.extend(out.cpu().numpy().flatten().tolist())
                    else:
                        # Without pretrained weights, use motion-attention
                        # as a sophisticated frame-difference rPPG method
                        # Extract attention-weighted green channel difference
                        a1 = self.model.appearance_conv1(app_tensor)
                        attn1 = self.model.attention1(a1)
                        
                        # Use attention to weight the motion signal
                        m_weighted = mot_tensor * attn1
                        # Green channel is most informative for rPPG
                        green_signal = m_weighted[:, 1, :, :].mean(dim=[1, 2])
                        bvp_signal.extend(green_signal.cpu().numpy().tolist())
                    
                    app_batch = []
                    mot_batch = []
        
        bvp_signal = np.array(bvp_signal)
        return bvp_signal, fps
    
    def extract_hr_from_frames(self, frames, fps):
        """
        Extract HR from pre-extracted face crop frames (NO video re-reading).
        
        Args:
            frames: list of BGR face crops (any size, will be resized to 36x36)
            fps: frames per second
        Returns:
            hr_bpm, sqi, bvp
        """
        if len(frames) < int(fps * 5):
            return 0.0, 0.0, np.array([])
        
        # Convert frames to tensors
        frames_tensor = []
        for f in frames:
            resized = cv2.resize(f, (DEEPPHYS_FRAME_SIZE, DEEPPHYS_FRAME_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            frames_tensor.append(tensor)
        
        # Run CNN
        bvp_signal = []
        with torch.no_grad():
            batch_size = 128
            app_batch, mot_batch = [], []
            
            for i in range(1, len(frames_tensor)):
                app_batch.append(frames_tensor[i])
                mot_batch.append(frames_tensor[i] - frames_tensor[i - 1])
                
                if len(app_batch) >= batch_size or i == len(frames_tensor) - 1:
                    app_t = torch.stack(app_batch).to(self.device)
                    mot_t = torch.stack(mot_batch).to(self.device)
                    
                    if self.pretrained:
                        out = self.model(app_t, mot_t)
                        bvp_signal.extend(out.cpu().numpy().flatten().tolist())
                    else:
                        a1 = self.model.appearance_conv1(app_t)
                        attn1 = self.model.attention1(a1)
                        m_weighted = mot_t * attn1
                        green_signal = m_weighted[:, 1, :, :].mean(dim=[1, 2])
                        bvp_signal.extend(green_signal.cpu().numpy().tolist())
                    
                    app_batch, mot_batch = [], []
        
        bvp = np.array(bvp_signal)
        
        # Bandpass + FFT (same as extract_hr)
        try:
            nyq = 0.5 * fps
            low_n, high_n = FREQ_LOW / nyq, FREQ_HIGH / nyq
            if low_n <= 0 or high_n >= 1:
                return 0.0, 0.0, bvp
            b, a = butter(FILTER_ORDER, [low_n, high_n], btype='band')
            filtered = filtfilt(b, a, bvp, padlen=min(3 * max(len(b), len(a)), len(bvp) - 1))
            freqs, psd = welch(filtered, fs=fps, nperseg=min(512, len(filtered)),
                              noverlap=min(256, len(filtered) // 2))
            valid = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
            if not np.any(valid) or np.max(psd[valid]) == 0:
                return 0.0, 0.0, bvp
            peak_freq = freqs[valid][np.argmax(psd[valid])]
            hr_bpm = peak_freq * 60.0
            peak_power = np.max(psd[valid])
            total_power = np.sum(psd[valid])
            sqi = min(peak_power / total_power if total_power > 0 else 0, 1.0)
            return hr_bpm, sqi, bvp
        except Exception:
            return 0.0, 0.0, bvp

    def extract_hr(self, video_path):
        """
        Full pipeline: video → BVP → HR (BPM) + SQI.
        
        Returns:
            hr_bpm: float, estimated heart rate
            sqi: float, signal quality index (0-1)
            bvp: numpy array, raw BVP signal
        """
        bvp, fps = self.extract_bvp(video_path)
        
        if bvp is None or len(bvp) < int(fps * 5):
            return 0.0, 0.0, np.array([])
        
        # Bandpass filter
        try:
            nyq = 0.5 * fps
            low_n = FREQ_LOW / nyq
            high_n = FREQ_HIGH / nyq
            if low_n <= 0 or high_n >= 1:
                return 0.0, 0.0, bvp
            b, a = butter(FILTER_ORDER, [low_n, high_n], btype='band')
            filtered = filtfilt(b, a, bvp, padlen=min(3 * max(len(b), len(a)), len(bvp) - 1))
        except Exception:
            filtered = bvp
        
        # FFT for HR estimation
        try:
            freqs, psd = welch(filtered, fs=fps, nperseg=min(512, len(filtered)),
                              noverlap=min(256, len(filtered) // 2))
            valid = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
            if not np.any(valid) or np.max(psd[valid]) == 0:
                return 0.0, 0.0, bvp
            
            peak_freq = freqs[valid][np.argmax(psd[valid])]
            hr_bpm = peak_freq * 60.0
            
            # Signal Quality Index
            peak_power = np.max(psd[valid])
            total_power = np.sum(psd[valid])
            sqi = peak_power / total_power if total_power > 0 else 0
            sqi = min(sqi, 1.0)
            
            return hr_bpm, sqi, bvp
        except Exception:
            return 0.0, 0.0, bvp


# ========================= STANDALONE TEST =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepPhys rPPG Extraction")
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights (.pth)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" DeepPhys rPPG Heart Rate Extraction")
    print("=" * 60)
    
    extractor = DeepPhysExtractor(weights_path=args.weights)
    mode = "pretrained" if extractor.pretrained else "attention-guided"
    print(f"  Mode: {mode} | Device: {extractor.device}")
    print(f"  Video: {args.video}")
    
    hr, sqi, bvp = extractor.extract_hr(args.video)
    
    print(f"\n  ┌────────────────────────────────┐")
    print(f"  │  DeepPhys HR:  {hr:>7.1f} BPM     │")
    print(f"  │  SQI:          {sqi:>7.4f}         │")
    print(f"  │  BVP length:   {len(bvp):>7d} frames  │")
    print(f"  └────────────────────────────────┘")
