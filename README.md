# Student Engagement Recognition System 🎓👁️
> **ADVITIYA | NxtGen Hackathon 2026 — AI for Education**
> *CDAC AI Impact Summit 2026 · IIT Ropar Pre-Summit Activity*

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

## 📖 Overview
This project addresses the challenge of measuring student attentiveness in online learning environments. Using Computer Vision and Deep Learning, the system analyses video feeds to classify student engagement levels, providing actionable insights for educators.

**Problem Statement:** Student Engagement Recognition using Visual and rPPG Information.

---

## 📂 Submission Structure (Judge-Facing)

```
analyzer/
├── README.md                   # This file
├── requirements.txt            # All dependencies
│
├── task1_2_visual/             # ── Visual-based engagement recognition ──
│   ├── train.py                # Self-contained training  (--task 1 | --task 2)
│   ├── inference.py            # Self-contained inference (--task 1 | --task 2)
│   └── model.pth               # Saved after training
│
├── task3_rppg/                 # ── rPPG-based engagement recognition ──
│   ├── train.py                # Placeholder / TODO
│   ├── inference.py            # Placeholder / TODO
│   └── model.pth               # Saved after training
│
└── dataset/
    ├── train/                  # Place training videos + labels here
    └── test/                   # Place test videos + labels here
```

> **Note:** Development files (`src/`, `scripts/`, `data/`, `models/`, `notebooks/`) are also present in the repo for reference but are **not** part of the judge submission.

---

## 🛠️ Installation

```bash
git clone <repo_url>
cd analyzer
pip install -r requirements.txt
```

Requires **Python 3.8+** and a CUDA-capable GPU (recommended).

---

## ⚡ Quick Start — Task 1 & 2 (Visual)

### Training

```bash
# Task 1 — Binary (Engaged / Not-Engaged)
python task1_2_visual/train.py \
    --task 1 \
    --data_dir dataset/train \
    --labels  dataset/train/labels.xlsx \
    --epochs 50

# Task 2 — 4-class (Highly / Nominally / Disengaged / Distracted)
python task1_2_visual/train.py \
    --task 2 \
    --data_dir dataset/train \
    --labels  dataset/train/labels.xlsx \
    --epochs 50
```

The best model is saved to `task1_2_visual/model.pth`.

### Inference / Evaluation

```bash
# Task 1
python task1_2_visual/inference.py \
    --task 1 \
    --data_dir dataset/test \
    --labels  dataset/test/labels.xlsx \
    --model   task1_2_visual/model.pth

# Task 2
python task1_2_visual/inference.py \
    --task 2 \
    --data_dir dataset/test \
    --labels  dataset/test/labels.xlsx \
    --model   task1_2_visual/model.pth
```

**Reported Metrics:**
| Task | Metrics |
|------|---------|
| Task 1 (Binary) | Accuracy, F1-score, Confusion Matrix |
| Task 2 (4-class) | Accuracy, F1-macro, per-class accuracy, Confusion Matrix |

---

## ⚡ Quick Start — Task 3 (rPPG)

> *Placeholder — implementation in progress.*

```bash
python task3_rppg/train.py     --data_dir dataset/train ...
python task3_rppg/inference.py --data_dir dataset/test  ...
```

---

## 🧠 Methodology

### Visual Pipeline (Task 1 & 2)

1. **Face Extraction:** MediaPipe detects and crops faces at **6 FPS** (research-standard temporal resolution).
2. **Model:** **ResNet18** (ImageNet-pretrained) with selective fine-tuning (layer3 + layer4 + FC unfrozen).
3. **Regularisation:**
   - MixUp augmentation (α = 0.2 binary / 0.1 multi-class)
   - Dropout (0.5), Label Smoothing (0.15 / 0.1)
   - Class-weighted CrossEntropyLoss
4. **Optimiser:** AdamW with differential LR (backbone 5e-5 / head 5e-4) + CosineAnnealingLR.
5. **Inference:** Test-Time Augmentation (horizontal flip) + probability averaging across all frames per video.

### rPPG Pipeline (Task 3)

*Planned: CHROM / POS / DeepPhys via rPPG-Toolbox integration.*

---

## 📁 Development Files (in `backup/`)

All original development files have been moved to `backup/` to keep the submission clean:

```
backup/
  ├── scripts/                 # Dev training & evaluation scripts
  ├── src/                     # Modular source code (config, models, dataset)
  ├── data/                    # Raw videos & processed frames
  ├── models/                  # Dev model checkpoints (.pth)
  ├── notebooks/               # Exploration notebooks
  ├── results_task1.csv        # Previous evaluation results
  ├── results_validation.csv
  ├── validation_split.txt
  ├── Project_Status_Report.md
  ├── PS.md, Rules.md, Rules_Clarification.md
  └── 1.avi, 2.avi             # Sample test videos
```

---

## 🤝 Team
**Team Analyzer**
*   *Participating in IIT Ropar Advitiya Hackathon 2026*
*   **Track:** AI for Education

---

## 📜 License
This project is created for the *ADVITIYA | NxtGen Hackathon 2026 — CDAC AI Impact Summit*.
