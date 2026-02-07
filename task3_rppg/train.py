"""
=========================================================
 DeepPhys Self-Supervised Training
 
 Trains the DeepPhys CNN using POS rPPG signal as pseudo
 ground-truth (teacher-student / knowledge distillation).
 
 Why this works:
   - POS algorithm gives reliable BVP waveforms (math-based)
   - We train DeepPhys to predict the same BVP from face crops
   - Loss: Negative Pearson Correlation (standard in rPPG lit.)
   - After training, DeepPhys is a LEARNED 3rd algorithm
 
 Usage:
   python task3_rppg/train.py --video_dir data/raw/videos/Train
   python task3_rppg/train.py --video_dir data/raw/videos/Train --epochs 15 --lr 1e-4
=========================================================
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepphys import DeepPhysModel, DEEPPHYS_FRAME_SIZE, FREQ_LOW, FREQ_HIGH, FILTER_ORDER

# ========================= CONFIG =========================
CLIP_LEN = 300          # Frames per training clip (~10s at 30fps)
BATCH_SIZE = 64         # Batch size for CNN
MIN_FRAMES = 150        # Skip videos shorter than this
SEED = 42


# ========================= LOSS FUNCTION =========================
class NegPearsonLoss(nn.Module):
    """
    Negative Pearson Correlation Loss.
    
    Standard loss for rPPG training (Yu et al., PhysNet 2019).
    Maximizes correlation between predicted and target BVP signals.
    Range: -1 (perfect) to +1 (anti-correlated).
    """
    def forward(self, pred, target):
        # pred, target: (N,) 1D tensors
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        
        num = (pred_mean * target_mean).sum()
        den = torch.sqrt((pred_mean ** 2).sum() * (target_mean ** 2).sum() + 1e-8)
        
        pearson = num / den
        return 1.0 - pearson  # minimize → maximize correlation


# ========================= DATA LOADING =========================
def generate_pos_signal(rgb_signals, fps):
    """
    Generate POS rPPG signal from RGB traces.
    Same algorithm as extract_rppg.py but standalone.
    Wang et al., IEEE TBIOM 2017.
    """
    rgb = np.array(rgb_signals, dtype=np.float64)
    if len(rgb) < 30:
        return None
    
    # Temporal normalization
    mean_rgb = np.mean(rgb, axis=0, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1.0
    Cn = rgb / mean_rgb
    
    # POS projection
    S1 = Cn[:, 1] - Cn[:, 2]        # G - B
    S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R
    
    # Adaptive alpha with sliding window
    win_size = max(int(fps * 1.6), 32)
    H = np.zeros(len(S1))
    
    for start in range(0, len(S1) - win_size, win_size // 2):
        end = start + win_size
        s1_win = S1[start:end]
        s2_win = S2[start:end]
        std_s2 = np.std(s2_win)
        alpha = np.std(s1_win) / std_s2 if std_s2 > 1e-10 else 1.0
        H[start:end] += s1_win + alpha * s2_win
    
    # Bandpass filter
    try:
        nyq = 0.5 * fps
        low_n, high_n = FREQ_LOW / nyq, FREQ_HIGH / nyq
        if low_n > 0 and high_n < 1:
            b, a = butter(FILTER_ORDER, [low_n, high_n], btype='band')
            H = filtfilt(b, a, H, padlen=min(3 * max(len(b), len(a)), len(H) - 1))
    except Exception:
        pass
    
    # Normalize to [-1, 1]
    mx = np.max(np.abs(H)) + 1e-10
    H = H / mx
    
    return H.astype(np.float32)


def load_video_data(video_path, clip_len=CLIP_LEN):
    """
    Load video, extract face crops + RGB signals for POS.
    Returns list of (face_crops_tensors, pos_target) clips.
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60:
        fps = 30.0
    
    face_tensors = []
    rgb_signals = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # Face bounding box from landmarks
            xs = [l.x for l in lm]
            ys = [l.y for l in lm]
            x1 = max(0, int(min(xs) * img_w) - 10)
            y1 = max(0, int(min(ys) * img_h) - 10)
            x2 = min(img_w, int(max(xs) * img_w) + 10)
            y2 = min(img_h, int(max(ys) * img_h) + 10)
            
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (DEEPPHYS_FRAME_SIZE, DEEPPHYS_FRAME_SIZE))
            face_rgb_norm = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(face_rgb_norm).permute(2, 0, 1)  # (3, 36, 36)
            face_tensors.append(tensor)
            
            # Face crop mean RGB for POS
            mean_rgb = face_rgb_norm.mean(axis=(0, 1))  # (R, G, B) in [0,1]
            rgb_signals.append(mean_rgb * 255.0)  # scale back for POS
        else:
            # Repeat last
            if face_tensors:
                face_tensors.append(face_tensors[-1])
                rgb_signals.append(rgb_signals[-1])
    
    cap.release()
    face_mesh.close()
    
    if len(face_tensors) < MIN_FRAMES:
        return [], fps
    
    # Generate POS target from RGB
    pos_target = generate_pos_signal(rgb_signals, fps)
    if pos_target is None:
        return [], fps
    
    # Split into clips (overlapping by 50%)
    clips = []
    step = clip_len // 2
    for start in range(0, len(face_tensors) - clip_len, step):
        end = start + clip_len
        clip_tensors = face_tensors[start:end]
        clip_target = pos_target[start:end]
        clips.append((clip_tensors, clip_target))
    
    # Always include last clip if not too short
    if len(face_tensors) >= clip_len:
        clip_tensors = face_tensors[-clip_len:]
        clip_target = pos_target[-clip_len:]
        clips.append((clip_tensors, clip_target))
    elif not clips:
        # Video shorter than clip_len but longer than MIN_FRAMES → use entire video
        clips.append((face_tensors, pos_target))
    
    return clips, fps


# ========================= TRAINING =========================
def train_on_clip(model, optimizer, criterion, clip_tensors, clip_target, device):
    """
    Train DeepPhys on one clip. Returns loss value.
    
    Forward pass: for each frame pair (appearance=current, motion=diff),
    predict BVP value. Compute Neg-Pearson loss against POS target.
    """
    model.train()
    
    # Build appearance & motion pairs
    app_list, mot_list = [], []
    for i in range(1, len(clip_tensors)):
        app_list.append(clip_tensors[i])
        mot_list.append(clip_tensors[i] - clip_tensors[i - 1])
    
    target = torch.from_numpy(clip_target[1:]).float().to(device)  # skip first frame
    
    # Process in batches (memory-efficient)
    all_preds = []
    for b_start in range(0, len(app_list), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(app_list))
        app_batch = torch.stack(app_list[b_start:b_end]).to(device)
        mot_batch = torch.stack(mot_list[b_start:b_end]).to(device)
        
        pred = model(app_batch, mot_batch).squeeze(-1)  # (B,)
        all_preds.append(pred)
    
    predicted = torch.cat(all_preds)  # (T-1,)
    
    # Neg-Pearson loss
    loss = criterion(predicted, target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train DeepPhys (self-supervised with POS)")
    parser.add_argument('--video_dir', type=str, default='data/raw/videos/Train',
                        help="Directory with training videos")
    parser.add_argument('--output', type=str, default='task3_rppg/deepphys_weights.pth',
                        help="Output weights path")
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--limit', type=int, default=0, help="Max videos (0=all)")
    parser.add_argument('--clip_len', type=int, default=CLIP_LEN, help="Frames per clip")
    args = parser.parse_args()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(" DeepPhys Self-Supervised Training")
    print(" Teacher: POS algorithm (Wang et al. 2017)")
    print(" Loss: Negative Pearson Correlation")
    print("=" * 60)
    print(f"  Device:    {device}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  LR:        {args.lr}")
    print(f"  Clip len:  {args.clip_len}")
    print(f"  Output:    {args.output}")
    print()
    
    # Find videos
    VIDEO_EXTS = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm',
                  '.AVI', '.MP4', '.MOV', '.MKV', '.WMV', '.WEBM')
    videos = [os.path.join(args.video_dir, f) for f in sorted(os.listdir(args.video_dir))
              if os.path.splitext(f)[1] in VIDEO_EXTS]
    
    if args.limit > 0:
        videos = videos[:args.limit]
    
    print(f"  Found {len(videos)} training videos\n")
    
    if not videos:
        print("ERROR: No videos found!")
        return
    
    # ---- Phase 1: Load all clips ----
    print("Phase 1: Extracting face crops + POS targets...")
    all_clips = []
    for vi, vpath in enumerate(videos):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        print(f"  [{vi+1}/{len(videos)}] {vname}...", end=' ', flush=True)
        clips, fps = load_video_data(vpath, clip_len=args.clip_len)
        print(f"{len(clips)} clips (fps={fps:.0f})")
        all_clips.extend(clips)
    
    print(f"\n  Total training clips: {len(all_clips)}")
    if not all_clips:
        print("ERROR: No valid clips extracted!")
        return
    
    # ---- Phase 2: Train ----
    print(f"\nPhase 2: Training DeepPhys CNN...")
    model = DeepPhysModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = NegPearsonLoss()
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        random.shuffle(all_clips)
        epoch_losses = []
        
        for ci, (clip_tensors, clip_target) in enumerate(all_clips):
            loss = train_on_clip(model, optimizer, criterion,
                                clip_tensors, clip_target, device)
            epoch_losses.append(loss)
            
            if (ci + 1) % 10 == 0 or ci == len(all_clips) - 1:
                avg = np.mean(epoch_losses[-10:])
                print(f"  Epoch {epoch}/{args.epochs} | Clip {ci+1}/{len(all_clips)} | "
                      f"Loss: {loss:.4f} | Avg(10): {avg:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        epoch_avg = np.mean(epoch_losses)
        scheduler.step()
        
        print(f"  ── Epoch {epoch} complete | Avg Loss: {epoch_avg:.4f}")
        
        if epoch_avg < best_loss:
            best_loss = epoch_avg
            torch.save(model.state_dict(), args.output)
            print(f"  ★ Best model saved → {args.output} (loss={best_loss:.4f})")
    
    print(f"\n{'=' * 60}")
    print(f" Training complete!")
    print(f" Best loss: {best_loss:.4f}")
    print(f" Weights saved: {args.output}")
    print(f"{'=' * 60}")
    print(f"\nTo use trained DeepPhys:")
    print(f"  python task3_rppg/update_deepphys.py --video_dir {args.video_dir} "
          f"--results_dir task3_rppg/results --force --weights {args.output}")


if __name__ == "__main__":
    main()
