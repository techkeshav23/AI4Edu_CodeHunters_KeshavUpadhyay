"""
=========================================================
 Student Engagement Recognition - rPPG Inference Script
 Task 3: rPPG Signal Generation
 
 Run trained rPPG model on test videos to extract signals.
 
 Usage:
   python task3_rppg/inference.py --test_dir dataset/test --model task3_rppg/model.pth

 Metrics: MAE (BPM), Pearson Correlation
=========================================================
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="rPPG Inference (Task 3)")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test videos")
    parser.add_argument("--model", type=str, default="task3_rppg/model.pth",
                        help="Path to rPPG model")
    parser.add_argument("--output", type=str, default="rppg_test_signals.csv",
                        help="Output CSV for test rPPG signals")
    args = parser.parse_args()
    
    print("Task 3: rPPG Inference")
    print(f"Test directory: {args.test_dir}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("\n[TODO] Implement rPPG inference")

if __name__ == "__main__":
    main()
