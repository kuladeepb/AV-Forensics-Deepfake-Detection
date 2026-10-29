"""
preprocessing.py  –  AV-Forensics
Shared preprocessing used by both the Colab training notebook and backend inference.
All heavy imports are lazy so the module can be imported without a GPU.
"""

from __future__ import annotations
import io, logging, os, subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── MediaPipe face-mesh (lazy singleton) ───────────────────────────────────────
_mp_face_mesh = None

def _get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _mp_face_mesh


# Indices of mouth-region landmarks in MediaPipe 468-point face mesh
MOUTH_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
]


def _extract_mouth_crop(frame_bgr: np.ndarray, img_size: int) -> Optional[np.ndarray]:
    """Return (img_size, img_size) uint8 grayscale mouth crop, or None if no face found."""
    import mediapipe as mp
    face_mesh = _get_face_mesh()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in MOUTH_IDXS])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad = int(max(x2 - x1, y2 - y1) * 0.4)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size))
    return gray


def load_lip_frames(
    video_path: str,
    n_frames: int,
    img_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample `n_frames` evenly from the video, extract mouth crops.

    Returns
    -------
    frames : np.ndarray shape (T, 1, H, W) float32, normalized to [0,1]
             or None if fewer than n_frames//2 mouth crops found.
    raw_frames : np.ndarray shape (T, H_full, W_full, 3) uint8 BGR
                 (used by the backend for evidence frame thumbnails)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None, None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    crops, raws = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        raws.append(frame)
        crop = _extract_mouth_crop(frame, img_size)
        crops.append(crop)  # may be None
    cap.release()

    valid = [(c, r) for c, r in zip(crops, raws) if c is not None]
    if len(valid) < n_frames // 2:
        log.warning("Too few mouth crops in %s (%d/%d)", video_path, len(valid), n_frames)
        return None, None

    # Pad / truncate to exactly n_frames
    while len(valid) < n_frames:
        valid.append(valid[-1])
    valid = valid[:n_frames]

    crops_arr = np.stack([v[0] for v in valid], axis=0)   # (T, H, W)
    raws_arr  = np.stack([v[1] for v in valid], axis=0)   # (T, H_full, W_full, 3)

    crops_f = crops_arr[:, np.newaxis, :, :].astype(np.float32) / 255.0  # (T, 1, H, W)
    return crops_f, raws_arr


def load_audio_16k(
    video_path: str,
    audio_len: int,
    sr: int = 16_000,
) -> Optional[np.ndarray]:
    """
    Extract audio from video via ffmpeg → 16 kHz mono float32 array.
    Pad or trim to `audio_len` samples.
    Returns None if no audio track is found.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        "-f", "s16le", "pipe:1",
        "-loglevel", "error",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            log.warning("No audio in %s", video_path)
            return None
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning("ffmpeg error for %s: %s", video_path, e)
        return None

    if len(wav) == 0:
        return None

    # Pad or trim
    if len(wav) >= audio_len:
        wav = wav[:audio_len]
    else:
        wav = np.pad(wav, (0, audio_len - len(wav)))
    return wav


def preprocess_video(
    video_path: str,
    n_frames: int = 32,
    img_size: int = 96,
    audio_len: int = 64_000,
    sr: int = 16_000,
) -> dict:
    """
    Full preprocessing for one video.

    Returns dict with keys:
      frames      – np.ndarray (T, 1, H, W) float32  or None
      audio       – np.ndarray (audio_len,) float32  or None
      raw_frames  – np.ndarray (T, H, W, 3) uint8    or None
      ok          – bool  (False if frames or audio is None)
      error       – str   (reason for failure)
    """
    frames, raws = load_lip_frames(video_path, n_frames, img_size)
    audio = load_audio_16k(video_path, audio_len, sr)

    ok = frames is not None and audio is not None
    err = ""
    if frames is None:
        err += "No detectable face/mouth. "
    if audio is None:
        err += "No audio track."
    return dict(frames=frames, audio=audio, raw_frames=raws, ok=ok, error=err.strip())

