# AV-Forensics 🎭🔍

**Audio-Visual Lip-Sync Deepfake Detection** — Conference Paper 2 implementation.

> *Audio-Visual Lip-Sync Consistency Transformer for Cross-Dataset Deepfake Video Detection and Modality-Level Forgery Classification*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Phase 1 — Training in Google Colab](#4-phase-1--training-in-google-colab)
5. [Phase 2 — Backend (FastAPI)](#5-phase-2--backend-fastapi)
6. [Phase 3 — Frontend (React)](#6-phase-3--frontend-react)
7. [Phase 4 — Forensic Report](#7-phase-4--forensic-report)
8. [Phase 5 — Free Deployment](#8-phase-5--free-deployment)
9. [Testing](#9-testing)
10. [Limitations & Ethics](#10-limitations--ethics)

---

## 1. Project Overview

AV-Forensics takes an uploaded video and:
1. Decides **REAL or FAKE** using audio–lip-sync consistency.
2. Says **which modality** was manipulated (RVRA / RVFA / FVRA / FVFA).
3. Shows **when** suspicious activity occurs — sync timeline + evidence frames.
4. Produces a downloadable **professional forensic PDF**.

### Key design choices

| Decision | Value |
|---|---|
| Dataset | FakeAVCeleb (Kaggle: `shreyaty08/fakeavceleb`) |
| Audio encoder | `utter-project/mHuBERT-147` — swap in `ml/config.py` |
| Video encoder | Lip Video Transformer (custom, 256-dim) |
| Fusion | Cross-modal transformer (2 layers, 4 heads) |
| Backend | FastAPI + PyTorch CPU |
| Frontend | React + Vite + Tailwind + Recharts |
| Report | ReportLab |
| Free hosting | Frontend → Vercel/Netlify · Backend → Hugging Face Spaces (Docker) |

---

## 2. Architecture

```
Video
 ├── frames → MediaPipe → mouth crop 96×96 → LipVideoTransformer → (B, T, 256)
 └── audio  → ffmpeg 16kHz → mHuBERT-147 → weighted sum → Linear → (B, T, 256)
                                     ↓
                          CrossModalFusion (concat + TransformerEncoder)
                                     ↓
                                 mean-pool
                                     ↓
                    ┌────────────────┼────────────────┐
               head_bin          head_4          head_afake / head_vfake
             Real / Fake      RVRA/RVFA/         auxiliary
               (2-class)      FVRA/FVFA           (2-class each)
                              (4-class)
```

**Sync loss** — applied only to real (RVRA) pairs:
```
L_sync = (1 − cos_sim(v_aligned, a_aligned)) + max(0, cos_sim(v, a_shifted) − cos_sim(v, a) + margin)
```

**Total loss:**
```
L = CE_bin + CE_4 + 0.5*(CE_afake + CE_vfake) + 0.3*L_sync
```

---

## 3. Repository Layout

```
avforensics/
├── ml/
│   ├── config.py            # All constants (swap AUDIO_MODEL_NAME here)
│   ├── preprocessing.py     # Shared: MediaPipe lip crop + ffmpeg audio
│   ├── dataset.py           # AVDataset, identity-grouped split, class weights
│   ├── model.py             # LipVideoTransformer + AudioEncoder + CrossModalFusion
│   ├── train.py             # Training loop (AMP, grad accum, early stopping)
│   ├── evaluate.py          # Metrics, confusion matrices, ROC curve, curves
│   └── colab_train.ipynb    # Full Colab notebook (14 cells, end-to-end)
│
├── backend/
│   ├── main.py              # FastAPI app: /api/analyze, /api/report/{id}
│   ├── inference.py         # MOCK_MODE + real sliding-window inference
│   ├── report.py            # ReportLab PDF generator
│   ├── config.py            # Backend config (env vars)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       └── test_api.py      # pytest end-to-end + unit tests
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx   # Drag-drop upload, progress, wake-up state
│   │   │   ├── ResultPage.jsx   # Verdict, modality, sync chart, evidence gallery
│   │   │   └── AboutPage.jsx    # Methodology + limitations
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   └── Footer.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html           # SEO meta + Open Graph
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
└── README.md
```

---

## 4. Phase 1 — Training in Google Colab

### Prerequisites
- Google Colab account (free T4 GPU)
- Google Drive (for checkpoints)
- Kaggle account + API key

### Steps

**4.1 Upload code to Colab**
```bash
# Option A: clone from GitHub (after you push this repo)
!git clone https://github.com/your-org/avforensics /content/avforensics

# Option B: upload the ml/ folder manually via the Colab file browser
```

**4.2 Open the notebook**

Open `ml/colab_train.ipynb` in Colab.

Runtime → Change runtime type → **T4 GPU**.

**4.3 Run all cells in order**

| Cell | Purpose |
|---|---|
| 1 | Mount Google Drive |
| 2 | Install libraries + ffmpeg |
| 3 | Imports + config |
| 4 | Download & organise FakeAVCeleb from Kaggle |
| 5 | Dataset counts + integrity check |
| 6 | Visualise samples (frames, mouth crops, audio) |
| 7 | Cache preprocessed arrays to Drive (.npy) |
| 8 | Build AVDataset + DataLoaders |
| 9 | Build model + shape assertions |
| 10 | Train (AMP, grad accum, early stopping, Drive checkpoints) |
| 11 | Full evaluation — Task 1 & Task 2 metrics + plots |
| 12 | Export `model.pth` + `config.json` to Drive |
| 13 | Test on held-out examples |
| 14 | Manual upload test (single video) |

**4.4 Download outputs from Drive**
- `model.pth` — model weights (~300 MB for mHuBERT-147 variant)
- `config.json` — preprocessing + label config

> **Swapping the audio encoder:** Change `AUDIO_MODEL_NAME` in `ml/config.py`  
> from `"utter-project/mHuBERT-147"` to `"facebook/wav2vec2-xls-r-300m"`.  
> Re-run from Cell 2 onwards.

---

## 5. Phase 2 — Backend (FastAPI)

### Run locally (mock mode — no weights needed)

```bash
cd avforensics/backend
pip install -r requirements.txt

# Mock mode (default): MOCK_MODEL=1
uvicorn main:app --reload --port 7860
```

Test: `curl http://localhost:7860/health` → `{"status":"ok","mock_mode":true}`

### Run with real weights

```bash
# Copy trained files here
cp /path/to/model.pth  avforensics/backend/model.pth
cp /path/to/config.json avforensics/backend/config.json

# Disable mock mode
MOCK_MODEL=0 uvicorn main:app --port 7860
```

### API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/analyze` | Upload video → JSON result |
| GET | `/api/report/{case_id}` | Download PDF report |
| GET | `/api/result/{case_id}` | Re-fetch cached JSON result |

**POST /api/analyze** — multipart form, field `file`.  
Returns:
```json
{
  "case_id": "uuid",
  "verdict": "FAKE",
  "confidence": 0.91,
  "modality": {
    "class": "FVFA",
    "label": "Both Manipulated",
    "probabilities": {"RVRA": 0.02, "RVFA": 0.05, "FVRA": 0.03, "FVFA": 0.90},
    "audio_fake_prob": 0.95,
    "video_fake_prob": 0.93
  },
  "sync_timeline": [{"t": 0.0, "score": 0.21}, ...],
  "suspect_segments": [{"start": 1.2, "end": 3.8, "mean_score": 0.19}],
  "evidence_frames": [{"t": 2.1, "score": 0.14, "frame_b64": "...", "mouth_b64": "..."}]
}
```

**Environment variables** (all optional):

| Variable | Default | Description |
|---|---|---|
| `MOCK_MODEL` | `1` | `1` = demo mode, `0` = real inference |
| `MODEL_PATH` | `model.pth` | Path to trained weights |
| `CONFIG_PATH` | `config.json` | Path to config |
| `MAX_UPLOAD_MB` | `50` | Max video size |
| `MAX_DURATION_SEC` | `30` | Max video duration |
| `CACHE_TTL_SEC` | `3600` | Result cache lifetime (1 h) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## 6. Phase 3 — Frontend (React)

### Run locally

```bash
cd avforensics/frontend
pnpm install       # or: npm install
pnpm dev
```

Open http://localhost:5173.

Set `VITE_API_URL` env var to point at your backend:

```bash
# .env.local
VITE_API_URL=http://localhost:7860
```

### Build for production

```bash
pnpm build
# Output in dist/
```

---

## 7. Phase 4 — Forensic Report

The PDF is generated by `backend/report.py` using **ReportLab**.

Sections:
1. Case ID + timestamp
2. File name + SHA-256 hash
3. Verdict + confidence bar
4. Modality finding + class probabilities table
5. Lip-audio sync timeline chart (with suspect regions shaded)
6. Evidence frames grid (full frame + mouth crop + timestamp + score)
7. Methodology (model names, window settings, version)
8. Limitations disclaimer

> The report is generated on-demand at `GET /api/report/{case_id}`. It is cached until the 1-hour TTL expires.

---

## 8. Phase 5 — Free Deployment

### 8.1 Backend → Hugging Face Spaces (Docker, CPU)

> **Free tier (as of 2025):** Hugging Face Spaces CPU Basic — 2 vCPU, 16 GB RAM.  
> Cold-start after inactivity: ~30 s. No persistent storage for uploads (use `/tmp`).

**Steps:**

1. Create a new Space: [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free)
   - Visibility: Public

2. The Space repository is a separate Git repo. Clone it:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/av-forensics-api
   ```

3. Copy backend files into the Space repo:
   ```bash
   cp avforensics/backend/* av-forensics-api/
   ```

4. **Add model weights** (after Colab training):
   - Use HF Hub CLI: `huggingface-cli upload YOUR_USERNAME/av-forensics-api model.pth model.pth`
   - Or use Git LFS (files > 10 MB): `git lfs track "*.pth"` then `git add model.pth`

5. Set Space secrets (Settings → Variables and secrets):
   ```
   MOCK_MODEL = 0
   ```

6. Push to the Space:
   ```bash
   cd av-forensics-api
   git add .
   git commit -m "deploy backend"
   git push
   ```

7. Space URL: `https://YOUR_USERNAME-av-forensics-api.hf.space`

8. **Test:**
   ```bash
   curl https://YOUR_USERNAME-av-forensics-api.hf.space/health
   ```

---

### 8.2 Frontend → Vercel (static, free)

> **Free tier (as of 2025):** Vercel Hobby — unlimited static sites, 100 GB bandwidth/month.

**Steps:**

1. Push the full repo to GitHub.

2. Go to [vercel.com/new](https://vercel.com/new) → Import your GitHub repo.

3. Configure:
   - **Root directory:** `avforensics/frontend`
   - **Build command:** `pnpm build` (or `npm run build`)
   - **Output directory:** `dist`
   - **Environment variable:** `VITE_API_URL = https://YOUR_USERNAME-av-forensics-api.hf.space`

4. Click **Deploy**.

5. Your URL: `https://av-forensics.vercel.app` (or your custom domain).

---

### 8.3 Frontend → Netlify (alternative)

> **Free tier (as of 2025):** Netlify Free — 100 GB bandwidth/month, 300 build minutes/month.

```bash
cd avforensics/frontend
pnpm build

# Install Netlify CLI
npm install -g netlify-cli
netlify login
netlify deploy --prod --dir=dist
```

Set env var in Netlify dashboard: **Site settings → Environment variables**:
```
VITE_API_URL = https://YOUR_USERNAME-av-forensics-api.hf.space
```

---

### 8.4 Google Search Console (indexing)

1. Go to [search.google.com/search-console](https://search.google.com/search-console).
2. Add your Vercel/Netlify URL as a property.
3. Verify ownership — use the **HTML tag** method (add the meta tag to `index.html`).
4. Submit your sitemap: `https://your-domain/sitemap.xml`  
   (You can generate a simple sitemap.xml and place it in `frontend/public/`.)
5. Use the **URL Inspection** tool to request indexing of your homepage.

---

### 8.5 Switching from mock to real model

After Colab training and uploading `model.pth` + `config.json` to your HF Space:

1. In HF Space Settings → Secrets: set `MOCK_MODEL = 0`
2. Restart the Space (Factory Reboot).
3. The backend will load `model.pth` at startup (~30-60 s cold start).
4. Verify: `GET /health` should return `{"mock_mode": false}`.

---

## 9. Testing

### Backend tests (mock mode, no GPU)

```bash
cd avforensics/backend
pip install pytest httpx

# Run all tests
MOCK_MODEL=1 pytest tests/test_api.py -v
```

Tests cover:
- `/health` endpoint
- Invalid file type → 415
- File too large → 413
- Valid mock upload → correct JSON structure
- Evidence frame structure
- Result re-fetch
- PDF report generation + content check
- `load_audio_16k` shape (requires ffmpeg)

### ML shape assertions (offline, no data)

```bash
cd avforensics/ml
python model.py
# ✓ All shape assertions passed.
```

---

## 10. Limitations & Ethics

| Limitation | Details |
|---|---|
| **Probabilistic** | Confidence scores ≠ ground truth. High confidence can still be wrong. |
| **Dataset scope** | Trained on FakeAVCeleb (English speech, celebrity videos). Other languages/accents not validated. |
| **Compression** | Heavily compressed video degrades lip-crop quality and detection accuracy. |
| **Frame-level labels** | Sync timeline is derived from model internals, not annotated ground truth. Timestamps are indicative. |
| **Not legal proof** | Do NOT use this as sole evidence in legal, financial, or editorial decisions. |
| **Coverage** | Only audio-visual desynchronisation is detected. Other deepfake types (e.g., purely visual) are out of scope. |

---

## Citation

If you use this system in research, please cite:

```bibtex
@inproceedings{avforensics2025,
  title     = {Audio-Visual Lip-Sync Consistency Transformer for Cross-Dataset Deepfake Video Detection and Modality-Level Forgery Classification},
  author    = {Your Name},
  booktitle = {Proceedings of [Conference Name]},
  year      = {2025},
}
```

Dataset: FakeAVCeleb — Khalid, R. et al. (2021). *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset.*

---

*AV-Forensics — research prototype. Results are probabilistic and NOT legal proof.*

