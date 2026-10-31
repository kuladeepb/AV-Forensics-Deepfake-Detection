# AV-Forensics 🎭🔍

**Audio-Visual Lip-Sync Deepfake Detection System**

> *Audio-Visual Lip-Sync Consistency Transformer for Cross-Dataset Deepfake Video Detection and Modality-Level Forgery Classification*

---

## 🏆 Benchmark Performance & Results

Tested on real-world human speech and synthetic generative AI avatar datasets (18 Authentic recordings + 15 AI talking avatars):

| Metric | Result | Status |
|---|:---:|:---:|
| **Authentic Real Human Videos (`real_videos/`)** | **18 / 18 (100.0%)** | ✅ Passed |
| **Deepfake Talking Avatars (`videos/`)** | **15 / 15 (100.0%)** | ✅ Passed |
| **Total Test Suite Accuracy** | **33 / 33 (100.0%)** | ✅ Passed |
| **Macro F1-Score** | **1.00** | ✅ Optimal |

---

## 📚 Key Project Deliverables

- 📄 **[Comprehensive Technical Project Report (PROJECT_REPORT.md)](PROJECT_REPORT.md):** Complete documentation from project inception to completion, detailing mathematical formulation, architecture, root-cause analyses of output inversion, and empirical validation.
- 📓 **[Presentation Demo Notebook (AV_Forensics_Demo_and_Evaluation.ipynb)](AV_Forensics_Demo_and_Evaluation.ipynb):** Ready-made interactive Jupyter Notebook designed for demonstrating the full pipeline to professors and reviewers.
- 🚀 **[Benchmark Validation Script (backend/benchmark_test.py)](backend/benchmark_test.py):** One-command evaluation script to reproduce the 100% test benchmark on any machine.

---

## 🌟 What the System Does

1. **Binary Detection:** Accurately classifies input video as **REAL (Authentic)** or **FAKE (Manipulated)**.
2. **4-Class Modality Attribution:**
   - `RVRA`: Real Video, Real Audio (Authentic)
   - `RVFA`: Real Video, Fake Audio (Audio Deepfake / Voice Clone)
   - `FVRA`: Fake Video, Real Audio (Lip-sync / AI Avatar Deepfake)
   - `FVFA`: Fake Video, Fake Audio (Full Synthetic Generation)
3. **Temporal Lip-Audio Sync Timeline:** Computes per-frame cosine similarity between acoustic phonemes and visual mouth articulators, visually plotting desynchronization dips over time.
4. **Forensic Evidence & PDF Export:** Automatically captures suspect frames and compiles an audit-grade PDF forensic report with cryptographic SHA-256 hashes.

---

## 🛠️ Architecture Overview

```
Input Video (.mp4)
 ├── Visual Stream ──► MediaPipe (478 Landmarks) ──► Mouth ROI (96×96) ──► Lip Video Transformer ──► (B, 32, 256)
 │                                                                                                        │
 └── Audio Stream  ──► 16kHz Mono Resampling     ──► mHuBERT-147 SSL   ──► Linear Projection     ──► (B, 32, 256)
                                                                                                          │
                                           Cross-Modal Attention Fusion (2 Layers, 4 Heads) ◄─────────────┘
                                                              │
                     ┌────────────────────────────────────────┼────────────────────────────────────────┐
                     ▼                                        ▼                                        ▼
             Binary Forgery Head                   4-Class Modality Head                      Per-Frame Sync Head
             (Authentic vs Fake)                   (RVRA, RVFA, FVRA, FVFA)                   (Cosine Sim Timeline)
```

---

## 🚀 Quick Start Guide (Local Setup)

### 1. Clone Repository & Prerequisites
```bash
git clone https://github.com/kuladeepb/AV-Forensics-Deepfake-Detection.git
cd AV-Forensics-Deepfake-Detection
```
- Python 3.10+ (with PyTorch and FFmpeg installed)
- Node.js 18+ and npm
- Model weights: `backend/model.pth` and `backend/config.json`

### 2. Environment Configuration
Copy the provided environment templates:
```bash
# Backend environment setup
cd backend
copy .env.example .env

# Frontend environment setup
cd ../frontend
copy .env.example .env.local
```

### 3. Run the Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
- Health Check: `http://localhost:8000/health` → `{"status": "ok", "mock_mode": false}`
- Swagger API Docs: `http://localhost:8000/docs`

### 4. Run the Frontend (React + Vite)
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser to interact with the full dashboard!

---

## 🧪 Reproducing the Benchmark Evaluation

To evaluate the model across the test video dataset with one command:
```bash
cd backend
python benchmark_test.py
```
Or specify a custom evaluation directory:
```bash
python benchmark_test.py --test-dir "path/to/deepfakevideos"
```

---

## 🔒 Security & Git Hygiene

- All sensitive variables, API keys, and local environment files (`.env`, `.env.local`) are strictly excluded via `.gitignore`.
- Model weights (`*.pth`) and preprocessed dataset arrays (`*.npy`) are excluded from Git to comply with GitHub's file size policies.
- Checkpoints can be mounted locally or downloaded directly from Google Drive / Hugging Face Spaces.
