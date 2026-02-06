# INDIAN INSTITUTE OF TECHNOLOGY ROPAR
**ADVITIYA | NxtGen**
**Hackathon on AI for Social Good**
*(Pre-Summit Activity - India AI Impact Summit 2026)*

**School of Artificial Intelligence and Data Engineering**
**Indian Institute of Technology Ropar**

**Date:** 6-7 February 2026
**Venue:** Senate Hall and S. Ramanujan Block, IIT Ropar
**Program Coordinators:** Dr. Puneet Kumar, Dr. Abhinav Kumar, Dr. Abhilasha Nanda, Dr. Vandana Bharti and Dr. Santosh Kumar Vipparthi

---

# Problem Statement | AI for Education
**AI IMPACT SUMMIT | BHARAT 2026 INDIA | CDAC**

## Student Engagement Recognition using Visual and rPPG Information

[cite_start]With the rapid growth of online learning platforms, virtual classrooms, and massive open online learning, understanding and measuring student attentiveness has become a critical research problem[cite: 13]. [cite_start]Unlike traditional classrooms, where instructors can directly observe behavioral cues, online learning environments lack natural feedback mechanisms[cite: 14]. [cite_start]This has motivated the problem statement for education[cite: 15].

[cite_start]This problem contributes to education by addressing one of the most pressing challenges in digital learning: understanding and supporting students when direct human observation is not possible[cite: 16].

Solving this problem has direct benefits for students. [cite_start]For example, when a student shows signs of reduced focus or increased cognitive load, the system could introduce short breaks, simplify explanations, or provide additional examples[cite: 17]. [cite_start]From an institutional perspective, this technology supports educators by providing actionable learning analytics[cite: 18]. [cite_start]Teachers and content designers can analyze which parts of a lecture consistently lead to reduced attentiveness, confusion, or fatigue, and subsequently refine their teaching material[cite: 19].

[cite_start]**Engagement dataset download link** [cite: 20]
[cite_start]**Base paper:** https://arxiv.org/abs/1804.00858 [cite: 21]

*Figure 1: Examples of frames from our engagement database. [cite_start]Top to bottom rows show engagement intensity level: [0 (low)-3 (high)].* [cite: 22]

---

## Dataset Information
[cite_start]There are 74 Videos total in the Train folder which has the following labels[cite: 24].

| Class Labels | Number of Videos | Description |
| :--- | :--- | :--- |
| 0 | 7 | Distracted |
| 0.33 | 23 | Disengaged |
| 0.66 | 22 | Nominally Engaged |
| 1 | 22 | Highly Engaged |

[cite_start][cite: 25]

---

## Phase A: The Visual Baseline
[cite_start]**Goal:** Build a standard computer vision model to see how well we can predict attentive state just by looking at the student[cite: 27].

### Task 1: Visual Only Binary Classification
[cite_start]**Objective:** Classify the student's state into two simple categories: High Attentiveness (1) or Low Attentiveness (0)[cite: 29]. To check whether the student is attentive in the classroom or not. [cite_start]By noticing the behavioral cues[cite: 30].

* **Input:** Raw video frames. [cite_start]Dataset Link[cite: 31].
* [cite_start]**Allowed Features:** Visual cues only, such as eye contact, facial expressions, and head posture[cite: 32]. [cite_start]You can use Facial Landmarks, Gaze tracking, Head pose estimation, or standard CNN feature extractors (ResNet, EfficientNet)[cite: 33].
* **Output:** A binary label for the video segment. (For unimodal binary classification, we have to treat 0 and 0.33 as one class and 0.66 and 1 as the other class.) [cite_start]Upload the modal to get it checked by the standard test cases (Only 2 submissions per task will be given)[cite: 34].

### Task 2: Visual Only Multi-Class Classification
[cite_start]**Objective:** Classify the student's state into granular levels (e.g. Highly Engaged, Nominally Engaged, Disengaged, Distracted)[cite: 36].
[cite_start]**Note:** This is harder than binary classification as the boundaries between states are subtle[cite: 37].
[cite_start]**Output:** Predict [cite: 37]

---

## Phase B: The Physiological Layer
[cite_start]**Goal:** Extract hidden heart rate signals from the video using rPPG[cite: 39].

### rPPG (Remote Photoplethysmography)
* [cite_start]**Definition:** rPPG is a computer vision technique that detects blood volume changes in the microvascular tissue of the face[cite: 41].
* [cite_start]**How it works:** Every time your heart beats, blood flows to your face, causing a microscopic change in skin color (too small for the human eye to see, but visible to a camera)[cite: 42].
* [cite_start]**Why is it used:** Unlike facial expressions, which can be faked (e.g., a bored student forcing a smile), heart rate and heart rate variability (HRV) are involuntary physiological responses[cite: 43]. [cite_start]They provide an objective measure of mental state and stress[cite: 44].

### Task 3: rPPG Signal Generation
[cite_start]Your goal is to turn the video of a face into a 1D signal waveform that represents the heartbeat[cite: 46].
[cite_start]**Tool:** You are encouraged to use the rPPG-Toolbox (or similar libraries)[cite: 47].

**The Process:**
1.  [cite_start]**Face Detection:** Crop the face from the video in the dataset[cite: 48].
2.  [cite_start]**Signal Extraction:** Apply an algorithm to convert the color changes in the skin pixels into a waveform[cite: 49].
3.  [cite_start]**Algorithm Selection:** Implement any 3 of the following 7 standard algorithms to test which works best for this dataset after getting the results: CHROM, POS, DeepPhys, PhysNet, EfficientPhys, PhysFormer, TS-CAN[cite: 51].

[cite_start]**Deliverable:** A clean .csv or .json file containing the raw rPPG signal for the test videos[cite: 52].

---

## Phase C: rPPG Extraction and Multimodal Analysis
[cite_start]**Goal:** Turn the raw heart signal into meaningful numbers and combine them with the visual data[cite: 54].

### Task 4: Feature Extraction & Fusion Strategy
[cite_start]Now that you have the raw heart signal (from Task 3) and the visual features (from Task 1), you need to combine them[cite: 56].

**Step A: Feature Extraction (The Physiological Features)**
[cite_start]Don't just feed the raw wave into the model; extract meaningful statistics[cite: 57]:
* [cite_start]**HR Proxy:** An HR Proxy is a substitute signal that represents your pulse like a wave showing the rhythmic beat of your blood flow extracted from video frames[cite: 58].
* **Variability Proxy (HRV):** Calculate Heart Rate Variability. [cite_start]High HRV often indicates a relaxed, focused state; low HRV can indicate stress or cognitive load[cite: 59, 60].
* [cite_start]**SQI (Signal Quality Index):** A score telling the model how trustworthy the signal is (e.g., if the student is moving too much, the signal might be noise)[cite: 61].
* [cite_start]**Bandpower Features:** Analyze the frequency ($LF/HF$ ratio) to measure sympathetic nervous system activity (stress/focus)[cite: 62].

**Step B: Fusion Strategy**
[cite_start]As a way to combine two different senses to get a better result, for example how you use both your eyes and your ears to understand a conversation in a noisy room[cite: 63]. [cite_start]The teams will be tested on how you will mix the Visual info and the Heart info to produce results[cite: 64].
* [cite_start]**Early Fusion:** Concatenate visual vectors and rPPG vectors into one long vector before feeding it to the classifier[cite: 65].
* [cite_start]**Late Fusion:** Train two separate models (one Visual, one rPPG) and average their probability scores at the very end[cite: 66].

**Ablation Requirement:** You must include a short report/graph showing the performance difference. Example: Visual Only accuracy was 65%. [cite_start]Visual + rPPG accuracy was 72%[cite: 67, 68].

### Task 5: Multimodal Model Performance
[cite_start]**Goal:** Prove that "Two Modalities are Better Than One." [cite: 70]
This is the final exam. [cite_start]Train your combined model to predict engagement[cite: 71].

* [cite_start]**Task 5a (Binary):** Multimodal prediction of Engaged vs. Not Engaged[cite: 72].
* [cite_start]**Task 5b (Multi-class):** Multimodal prediction of the granular engagement levels[cite: 73].

[cite_start]**Success Metric:** Your Multimodal Model (Task 5) MUST statistically outperform your Visual-Only Baseline (Task 1 & 2)[cite: 74]. [cite_start]If it doesn't, revisit your fusion strategy or rPPG signal quality[cite: 75].

---

## Phase D: Bonus & Application
[cite_start]**Goal:** Take it out of the lab and into the real world[cite: 77].
[cite_start]**Note:** Try to solve it, proposals are also accepted! [cite: 78]

### Task 6: Demonstration & Deployment
Hybrid AI demonstrates Learning to Learn. [cite_start]Show that your model can adapt to a new student (whom it has never seen before) with only a few seconds of data, rather than needing to be retrained from scratch[cite: 80].
[cite_start]This solves the new user problem in educational software[cite: 81].

---

## Evaluation Details
[cite_start]**Evaluation Criteria:** The below mentioned are the metrics that are required to be included in the code files provided by the team[cite: 83].

| Phase | Task | Task Name | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :--- | :--- |
| A | Task 1 | Visual Binary Class | Accuracy | F1-Score |
| A | Task 2 | Visual Multi-Class | Accuracy (Class-wise) | F1(Macro), Confusion Matrix |
| B | Task 3 | rPPG Signal Generation | MAE (BPM) | Pearson Correlation |
| C | Task 4 | Multimodal Binary | Accuracy | Confusion Matrix |
| C | Task 5 | Multimodal Multi-Class | Accuracy, Loss | F1-Score (Macro), Higher Improvement over Task 2 |
| D | Task 6 | Deployment (Bonus) | Latency (ms) | UI Experience (Qualitative) |

[cite_start][cite: 84]

| PHASE (Task-wise) | Marks Percentage |
| :--- | :--- |
| Phase A (Task 1 & 2) | 30% |
| Phase B (Task 3) | 20% |
| Phase C (Task 4 & 5) | 30% |
| Phase D (Task 6) | 20% |

[cite_start][cite: 85]

## Qualification Criteria

| PHASE | QUALIFICATION CRITERIA (Accuracy) |
| :--- | :--- |
| PHASE A (Task 1: For Binary classification) | 70% |
| PHASE A (Task 2: For Multiclass classification) | 65% |
| PHASE B (Task 3: For rPPG) | 62% |
| PHASE D (Task 5: Multimodal) | 60% |

[cite_start][cite: 87]