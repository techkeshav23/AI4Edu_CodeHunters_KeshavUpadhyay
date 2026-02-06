# Project Analysis & Implementation Status Report

## Part 1: Problem Statement Verification (PS Analysis)

**Project Name:** Student Engagement Recognition (Multi-modal)
**Goal:** Classify if a student is "Attentive" or "Not Attentive" using Video (Visual) and Heart Rate (rPPG).

### Phase Breakdown

#### **Phase A: The Visual Baseline (Current Focus)**
*   **Task 1 (Binary):** Classify `Low Attention` (Labels 0, 0.33) vs `High Attention` (Labels 0.66, 1).
    *   **Success Metric:** Accuracy > **70%**.
*   **Task 2 (Multi-class):** 4 Classes (Distracted, Disengaged, Nominally Engaged, Highly Engaged).
    *   **Success Metric:** Accuracy > **65%**.
*   **Allowed Tech:** CNNs (ResNet/EfficientNet), Facial Landmarks, Gaze.

#### **Phase B: The Physiological Layer**
*   **Task 3:** Extract Heart Rate signals from face video using **rPPG**.
*   **Constraint:** Must implement **3 of 7** specific algorithms (e.g., CHROM, POS, PhysNet).

#### **Phase C: Multimodal Fusion**
*   **Task 4 & 5:** Combine Visual + rPPG features.
*   **Requirement:** Combined model **MUST** outperform Visual-only baseline.

---

## Part 2: Implementation Status (What is Done)

We have successfully set up the **Complete Pipeline for Phase A (Task 1)** on the C-DAC Server.

### 1. Infrastructure & Config `src/common/config.py`
*   **Status:** ✅ Done.
*   **Details:** Configured for Server Paths. Batch Size increased to **64** (optimized for A5000 GPU).

### 2. Preprocessing Engine `src/preprocessing/face_extractor.py` & `scripts/prepare_data.py`
*   **Status:** ✅ Done & Uploaded.
*   **Logic:**
    *   Reads `labels_train.xlsx`.
    *   Uses **MediaPipe** to detect faces.
    *   **Optimization:** Applied **"Tighter Crop" (Padding=0)** to focus on eyes/gaze and remove background noise (as per ML expert advice).
    *   Saves face images to `data/processed`.

### 3. Model Architecture `src/visual/models.py`
*   **Status:** ✅ Done.
*   **Architecture:** **ResNet18** (Pre-trained on ImageNet).
*   **Reasoning:** Chosen over ResNet50 to prevent overfitting on the small dataset (84 videos).

### 4. Training Pipeline `scripts/train_visual.py`
*   **Status:** ✅ Done & Uploaded.
*   **Critical Improvements Applied:**
    *   **Class Weights:** Applied `[1.5, 1.0]` weights to handle class imbalance (fewer Distracted videos).
    *   **Label Smoothing:** Set to `0.1` to handle noisy/weak labels.
    *   **Augmentation:** Random Flips and Rotations added.

### 5. Deployment
*   **Status:** ✅ Upload Complete.
*   **Next Action:** Running `python scripts/prepare_data.py` followed by `python scripts/train_visual.py` on the SSH terminal.

---

## Missing / To-Do (Next Steps)
1.  **Phase B (rPPG):** Code not yet written (Planned: POS, CHROM, PhysNet).
2.  **Phase C (Fusion):** Dependent on Phase A & B completion.
