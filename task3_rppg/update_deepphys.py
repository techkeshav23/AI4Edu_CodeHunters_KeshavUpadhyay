"""
=========================================================
 Update existing results with DeepPhys (3rd algorithm)
 
 Reads existing .npy files (POS + CHROM already done),
 runs ONLY DeepPhys on each video, updates .npy + summary.csv.
 
 Usage:
   python task3_rppg/update_deepphys.py --video_dir data/raw/videos/Train --results_dir task3_rppg/results
   python task3_rppg/update_deepphys.py --video_dir data/raw/videos/Train --results_dir task3_rppg/results --limit 5
=========================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepphys import DeepPhysExtractor

VIDEO_EXTS = ('.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm')


def main():
    parser = argparse.ArgumentParser(description="Add DeepPhys to existing rPPG results")
    parser.add_argument('--video_dir', type=str, required=True, help="Video directory")
    parser.add_argument('--results_dir', type=str, default='task3_rppg/results', help="Results dir with .npy files")
    parser.add_argument('--limit', type=int, default=0, help="Max videos (0=all)")
    parser.add_argument('--weights', type=str, default=None, help="DeepPhys weights path")
    parser.add_argument('--force', action='store_true', help="Re-run even if DeepPhys already done")
    args = parser.parse_args()

    print("=" * 60)
    print(" DeepPhys Update — Adding 3rd algorithm to existing results")
    print("=" * 60)

    # Init DeepPhys once
    extractor = DeepPhysExtractor(weights_path=args.weights)
    mode = "pretrained" if extractor.pretrained else "attention-guided"
    print(f"  Mode: {mode} | Device: {extractor.device}\n")

    # Find all existing .npy results
    npy_files = sorted([f for f in os.listdir(args.results_dir) if f.endswith('_rppg.npy')])
    print(f"Found {len(npy_files)} existing .npy results in {args.results_dir}")

    if args.limit > 0:
        npy_files = npy_files[:args.limit]
        print(f"Processing first {args.limit} only")

    updated = 0
    skipped = 0
    failed = 0
    all_results = []

    for i, npy_name in enumerate(npy_files):
        video_name = npy_name.replace('_rppg.npy', '')
        npy_path = os.path.join(args.results_dir, npy_name)

        # Load existing data
        try:
            data = np.load(npy_path, allow_pickle=True).item()
        except Exception as e:
            print(f"  [{i+1}/{len(npy_files)}] {video_name} — ERROR loading .npy: {e}")
            failed += 1
            continue

        # Skip if DeepPhys already done (unless --force)
        if not args.force and data.get('hr_deepphys', 0) > 0:
            print(f"  [{i+1}/{len(npy_files)}] {video_name} — already has DeepPhys ({data['hr_deepphys']:.1f} BPM), skip")
            all_results.append(data_to_row(data, video_name))
            skipped += 1
            continue

        # Find video file
        video_path = find_video(args.video_dir, video_name)
        if video_path is None:
            print(f"  [{i+1}/{len(npy_files)}] {video_name} — video not found, skip")
            data.setdefault('hr_deepphys', 0.0)
            data.setdefault('sqi_deepphys', 0.0)
            all_results.append(data_to_row(data, video_name))
            skipped += 1
            continue

        # Run DeepPhys only
        print(f"  [{i+1}/{len(npy_files)}] {video_name}...", end=' ', flush=True)
        try:
            hr_deep, sqi_deep, _ = extractor.extract_hr(video_path)
            data['hr_deepphys'] = hr_deep
            data['sqi_deepphys'] = sqi_deep
            np.save(npy_path, data)
            print(f"DeepPhys: {hr_deep:.1f} BPM (SQI: {sqi_deep:.4f}) ✓")
            updated += 1
        except Exception as e:
            print(f"ERROR: {e}")
            data['hr_deepphys'] = 0.0
            data['sqi_deepphys'] = 0.0
            failed += 1

        all_results.append(data_to_row(data, video_name))

    # Regenerate summary.csv with all 3 algorithms
    summary_path = os.path.join(args.results_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'fps', 'frames', 'hr_pos', 'hr_chrom', 'hr_deepphys', 'sqi_pos', 'sqi_chrom', 'sqi_deepphys'])
        for r in all_results:
            writer.writerow([
                r['video'], r['fps'], r['frames'],
                round(r['hr_pos'], 2), round(r['hr_chrom'], 2), round(r.get('hr_deepphys', 0), 2),
                round(r['sqi_pos'], 4), round(r['sqi_chrom'], 4), round(r.get('sqi_deepphys', 0), 4)
            ])

    print(f"\n{'=' * 60}")
    print(f" DONE: Updated {updated} | Skipped {skipped} | Failed {failed}")
    print(f" Summary saved: {summary_path}")
    print(f"{'=' * 60}")


def data_to_row(data, video_name):
    return {
        'video': video_name,
        'fps': data.get('fps', 30),
        'frames': data.get('frames', 0),
        'hr_pos': data.get('hr_pos', 0),
        'hr_chrom': data.get('hr_chrom', 0),
        'hr_deepphys': data.get('hr_deepphys', 0),
        'sqi_pos': data.get('sqi_pos', 0),
        'sqi_chrom': data.get('sqi_chrom', 0),
        'sqi_deepphys': data.get('sqi_deepphys', 0),
    }


def find_video(video_dir, video_name):
    """Find video file matching name with any extension."""
    for ext in VIDEO_EXTS:
        # Try exact case
        path = os.path.join(video_dir, video_name + ext)
        if os.path.exists(path):
            return path
        # Try uppercase
        path = os.path.join(video_dir, video_name + ext.upper())
        if os.path.exists(path):
            return path
    # Fallback: scan directory
    for f in os.listdir(video_dir):
        name_no_ext = os.path.splitext(f)[0]
        if name_no_ext == video_name:
            return os.path.join(video_dir, f)
    return None


if __name__ == "__main__":
    main()
