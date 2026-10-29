"""
inference.py  –  AV-Forensics Backend
Loads model + config, runs sliding-window inference, extracts evidence.
In MOCK_MODE (MOCK_MODEL=1) returns realistic fake data so the frontend
and report can be built before trained weights exist.
"""

from __future__ import annotations
import base64, hashlib, json, logging, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Resolve ml/ on path so we can import shared code ──────────────────────────
_THIS = Path(__file__).parent
_ML   = _THIS.parent / "ml"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

try:
    from backend_config import (
        MOCK_MODEL, MODEL_PATH, CONFIG_PATH,
        TOP_EVIDENCE_N, MIN_FRAME_GAP,
        WINDOW_SEC, STRIDE_SEC,
    )
except ImportError:
    from config import (
        MOCK_MODEL, MODEL_PATH, CONFIG_PATH,
        TOP_EVIDENCE_N, MIN_FRAME_GAP,
        WINDOW_SEC, STRIDE_SEC,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK MODE
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_result(filename: str, sha256: str, duration: float) -> dict:
    """Return realistic-looking fake data for frontend/report development."""
    import random, uuid
    rng = random.Random(sha256[:8])
    is_fake   = rng.random() > 0.35
    conf      = rng.uniform(0.72, 0.97) if is_fake else rng.uniform(0.68, 0.94)
    cls_probs = np.array([rng.random() for _ in range(4)])
    cls_probs = (cls_probs / cls_probs.sum()).tolist()
    cls_idx   = (0 if not is_fake else rng.choice([1, 2, 3]))
    CLASSES   = ["RVRA", "RVFA", "FVRA", "FVFA"]
    MODALITY_LABELS = {
        "RVRA": "Authentic",
        "RVFA": "Audio Manipulated",
        "FVRA": "Video Manipulated",
        "FVFA": "Both Manipulated",
    }

    n_steps = max(8, int(duration / 0.5))
    sync_timeline = []
    for i in range(n_steps):
        t = i * (duration / n_steps)
        base = 0.75 if not is_fake else 0.35
        score = float(np.clip(base + rng.uniform(-0.2, 0.2), 0, 1))
        sync_timeline.append({"t": round(t, 2), "score": round(score, 4)})

    # Suspect segments = windows where sync is low
    suspects = []
    if is_fake:
        for j in range(0, n_steps - 3, 4):
            start = sync_timeline[j]["t"]
            end   = sync_timeline[min(j + 3, n_steps - 1)]["t"]
            mean_s = np.mean([sync_timeline[k]["score"] for k in range(j, min(j + 4, n_steps))])
            if mean_s < 0.5:
                suspects.append({"start": round(start, 2), "end": round(end, 2), "mean_score": round(float(mean_s), 4)})
    suspects = suspects[:3]

    # Evidence frames: solid-color placeholder + base64
    evidence_frames = []
    for k in range(min(TOP_EVIDENCE_N, 4)):
        t_ev = sync_timeline[k * (n_steps // max(TOP_EVIDENCE_N, 1))]["t"] if n_steps > 0 else k
        score_ev = sync_timeline[k * (n_steps // max(TOP_EVIDENCE_N, 1))]["score"] if n_steps > 0 else 0.5
        # Tiny placeholder gray image
        img = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.putText(img, f"t={t_ev:.1f}", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        _, buf = cv2.imencode(".jpg", img)
        b64  = base64.b64encode(buf).decode()
        mouth = np.full((32, 32, 3), 80, dtype=np.uint8)
        _, m_buf = cv2.imencode(".jpg", mouth)
        m_b64 = base64.b64encode(m_buf).decode()
        evidence_frames.append({
            "t":         round(float(t_ev), 2),
            "score":     round(float(score_ev), 4),
            "frame_b64": b64,
            "mouth_b64": m_b64,
        })

    return {
        "verdict":    "FAKE" if is_fake else "REAL",
        "confidence": round(conf, 4),
        "modality": {
            "class":          CLASSES[cls_idx],
            "label":          MODALITY_LABELS[CLASSES[cls_idx]],
            "probabilities":  {c: round(p, 4) for c, p in zip(CLASSES, cls_probs)},
            "audio_fake_prob": round(cls_probs[1] + cls_probs[3], 4),
            "video_fake_prob": round(cls_probs[2] + cls_probs[3], 4),
        },
        "sync_timeline":    sync_timeline,
        "suspect_segments": suspects,
        "evidence_frames":  evidence_frames,
        "_mock": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REAL INFERENCE MODE
# ═══════════════════════════════════════════════════════════════════════════════

class _ModelWrapper:
    """Singleton: load once, reuse across requests."""
    _instance: Optional["_ModelWrapper"] = None

    @classmethod
    def get(cls) -> "_ModelWrapper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import torch
        from ml_config import (  # type: ignore  (imported from ml/ at runtime)
            AUDIO_MODEL_NAME, N_FRAMES, IMG_SIZE, AUDIO_LEN, SR,
            EMBED_DIM, FAKE_THRESHOLD, CLASSES, BINARY_LABEL,
            AUDIO_FAKE, VIDEO_FAKE, MODALITY_LABELS,
        )
        from model import AVDeepfakeModel  # type: ignore

        with open(CONFIG_PATH) as f:
            cfg = json.load(f)

        self.cfg        = cfg
        self.device     = torch.device("cpu")
        self.n_frames   = cfg.get("n_frames",   N_FRAMES)
        self.img_size   = cfg.get("img_size",   IMG_SIZE)
        self.audio_len  = cfg.get("audio_len",  AUDIO_LEN)
        self.sr         = cfg.get("sr",         SR)
        self.threshold  = cfg.get("fake_threshold", FAKE_THRESHOLD)
        self.classes    = cfg.get("classes",    CLASSES)
        self.binary_label = cfg.get("binary_label", BINARY_LABEL)
        self.audio_fake = cfg.get("audio_fake", AUDIO_FAKE)
        self.video_fake = cfg.get("video_fake", VIDEO_FAKE)
        audio_model = cfg.get("audio_model_name", AUDIO_MODEL_NAME)

        self.model = AVDeepfakeModel(
            embed_dim=cfg.get("embed_dim", EMBED_DIM),
            audio_model=audio_model,
            freeze_cnn=False,
            freeze_layers=0,
        ).to(self.device)

        state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        log.info("Model loaded from %s", MODEL_PATH)

    def predict_window(
        self,
        frames: np.ndarray,    # (T, 1, H, W)
        audio:  np.ndarray,    # (audio_len,)
    ) -> dict:
        import torch, torch.nn.functional as F
        with torch.no_grad():
            f = torch.from_numpy(frames).unsqueeze(0).to(self.device)
            a = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            out = self.model(f, a)

            prob_bin  = F.softmax(out["logits_bin"], 1)[0]
            prob_4    = F.softmax(out["logits_4"],   1)[0]
            prob_af   = F.softmax(out["logits_afake"], 1)[0, 1].item()
            prob_vf   = F.softmax(out["logits_vfake"], 1)[0, 1].item()

            v_n = F.normalize(out["vis_emb"][0], dim=-1)
            a_n = F.normalize(out["aud_emb"][0], dim=-1)
            sync = (v_n * a_n).sum(-1).cpu().numpy()

        return {
            "fake_prob":   prob_bin[1].item(),
            "prob_4":      prob_4.cpu().numpy(),
            "audio_fake":  prob_af,
            "video_fake":  prob_vf,
            "sync":        sync,       # (T,)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence frame extraction helper
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_to_b64(frame_bgr: np.ndarray, max_dim: int = 320) -> str:
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def _extract_mouth_b64(frame_bgr: np.ndarray, img_size: int = 96) -> str:
    """Try to extract mouth crop; fall back to full frame if no face found."""
    try:
        from preprocessing import _extract_mouth_crop  # type: ignore
        crop = _extract_mouth_crop(frame_bgr, img_size)
        if crop is not None:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf).decode()
    except Exception:
        pass
    return _frame_to_b64(frame_bgr, max_dim=img_size)


# ═══════════════════════════════════════════════════════════════════════════════
# Main analysis function
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_video(video_path: str, filename: str) -> dict:
    """
    Full analysis pipeline.
    Returns the JSON response dict consumed by /api/analyze.
    """
    import subprocess

    # Compute SHA-256
    sha256 = hashlib.sha256(Path(video_path).read_bytes()).hexdigest()

    # Get duration via ffprobe
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15,
        )
        duration = float(proc.stdout.strip() or "10")
    except Exception:
        duration = 10.0

    if MOCK_MODEL:
        result = _mock_result(filename, sha256, duration)
        result.update({"sha256": sha256, "duration": round(duration, 2), "filename": filename})
        return result

    # ── Real inference ────────────────────────────────────────────────────────
    from preprocessing import load_lip_frames, load_audio_16k  # type: ignore

    wrapper = _ModelWrapper.get()
    n_frames  = wrapper.n_frames
    img_size  = wrapper.img_size
    audio_len = wrapper.audio_len
    sr        = wrapper.sr
    classes   = wrapper.classes
    MODALITY_LABELS = {
        "RVRA": "Authentic",
        "RVFA": "Audio Manipulated",
        "FVRA": "Video Manipulated",
        "FVFA": "Both Manipulated",
    }

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    window_frames = int(WINDOW_SEC * fps)
    stride_frames = int(STRIDE_SEC * fps)

    # Sliding window
    all_fake_prob : List[float] = []
    all_prob_4    : List[np.ndarray] = []
    all_audio_fake: List[float] = []
    all_video_fake: List[float] = []
    all_sync      : List[Tuple[float, float]] = []  # (t, score)

    step = 0
    while True:
        t_start = step * STRIDE_SEC
        f_start = step * stride_frames
        if f_start >= n_total:
            break

        # Extract window frames
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        raw_frames, mouth_crops = [], []

        indices = np.linspace(0, window_frames - 1, n_frames, dtype=int)
        frame_buf = []
        for _ in range(window_frames):
            ok, frm = cap.read()
            if ok:
                frame_buf.append(frm)
        cap.release()

        from preprocessing import _extract_mouth_crop  # type: ignore
        for idx in indices:
            if idx < len(frame_buf):
                frm = frame_buf[idx]
            else:
                frm = frame_buf[-1] if frame_buf else np.zeros((224, 224, 3), np.uint8)
            raw_frames.append(frm)
            crop = _extract_mouth_crop(frm, img_size)
            mouth_crops.append(crop)

        # Fill missing
        last_valid = next((c for c in mouth_crops if c is not None), np.zeros((img_size, img_size), np.uint8))
        crops_arr = np.stack(
            [c if c is not None else last_valid for c in mouth_crops],
            axis=0,
        )[:, np.newaxis].astype(np.float32) / 255.0  # (T, 1, H, W)

        # Audio slice
        full_wav = load_audio_16k(video_path, sr * int(duration + 2), sr)
        if full_wav is None:
            full_wav = np.zeros(audio_len, np.float32)
        a_start = int(t_start * sr)
        a_end   = a_start + audio_len
        audio_slice = full_wav[a_start:a_end] if a_end <= len(full_wav) else \
                      np.pad(full_wav[a_start:], (0, max(0, a_end - len(full_wav))))
        audio_slice = audio_slice[:audio_len]
        if len(audio_slice) < audio_len:
            audio_slice = np.pad(audio_slice, (0, audio_len - len(audio_slice)))

        pred = wrapper.predict_window(crops_arr, audio_slice)
        all_fake_prob.append(pred["fake_prob"])
        all_prob_4.append(pred["prob_4"])
        all_audio_fake.append(pred["audio_fake"])
        all_video_fake.append(pred["video_fake"])

        T = len(pred["sync"])
        for ti, sc in enumerate(pred["sync"]):
            t_abs = t_start + ti * (WINDOW_SEC / T)
            all_sync.append((round(t_abs, 2), round(float(sc), 4)))

        step += 1

    if not all_fake_prob:
        raise ValueError("Could not process any window from the video.")

    # Aggregate
    mean_fake   = float(np.mean(all_fake_prob))
    mean_prob_4 = np.mean(all_prob_4, axis=0)
    mean_af     = float(np.mean(all_audio_fake))
    mean_vf     = float(np.mean(all_video_fake))

    verdict     = "FAKE" if mean_fake > wrapper.threshold else "REAL"
    confidence  = round(mean_fake if verdict == "FAKE" else 1 - mean_fake, 4)
    cls_idx     = int(np.argmax(mean_prob_4))
    pred_cls    = classes[cls_idx]

    # Sync timeline (deduplicate times)
    seen_t = set()
    sync_timeline = []
    for t, sc in sorted(all_sync):
        if t not in seen_t:
            seen_t.add(t)
            sync_timeline.append({"t": t, "score": sc})

    # Suspect segments (low sync)
    _scores = [s["score"] for s in sync_timeline]
    threshold_sync = 0.5
    suspects: List[dict] = []
    in_seg, seg_start, seg_scores = False, 0.0, []
    for entry in sync_timeline:
        if entry["score"] < threshold_sync:
            if not in_seg:
                in_seg = True; seg_start = entry["t"]; seg_scores = []
            seg_scores.append(entry["score"])
        else:
            if in_seg:
                suspects.append({
                    "start": seg_start,
                    "end":   entry["t"],
                    "mean_score": round(float(np.mean(seg_scores)), 4),
                })
                in_seg = False
    suspects = suspects[:3]

    # Evidence frames: lowest-sync moments
    sorted_sync = sorted(sync_timeline, key=lambda x: x["score"])
    evidence_frames: List[dict] = []
    last_t = -999.0
    for entry in sorted_sync:
        if abs(entry["t"] - last_t) < (MIN_FRAME_GAP / fps):
            continue
        last_t = entry["t"]

        # Seek to frame
        f_idx = int(entry["t"] * fps)
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok2, ev_frame = cap2.read()
        cap2.release()
        if not ok2:
            continue

        evidence_frames.append({
            "t":         entry["t"],
            "score":     entry["score"],
            "frame_b64": _frame_to_b64(ev_frame),
            "mouth_b64": _extract_mouth_b64(ev_frame, img_size),
        })
        if len(evidence_frames) >= TOP_EVIDENCE_N:
            break

    return {
        "filename":   filename,
        "sha256":     sha256,
        "duration":   round(duration, 2),
        "verdict":    verdict,
        "confidence": confidence,
        "modality": {
            "class":          pred_cls,
            "label":          MODALITY_LABELS.get(pred_cls, pred_cls),
            "probabilities":  {c: round(float(p), 4) for c, p in zip(classes, mean_prob_4)},
            "audio_fake_prob": round(mean_af, 4),
            "video_fake_prob": round(mean_vf, 4),
        },
        "sync_timeline":    sync_timeline,
        "suspect_segments": suspects,
        "evidence_frames":  evidence_frames,
        "_mock": False,
    }

