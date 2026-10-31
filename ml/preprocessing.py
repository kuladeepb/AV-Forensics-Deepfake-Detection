"""
preprocessing.py  –  AV-Forensics
State-of-the-art lip cropping using Google MediaPipe FaceLandmarker with 468 3D landmarks.
Works seamlessly across modern Python (including Python 3.13) and NumPy 2.
"""

from __future__ import annotations
import io, logging, os, subprocess, urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Official 30 lip contour landmark indices in MediaPipe 468-point mesh
MOUTH_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
]

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_path = "/content/face_landmarker.task"
            if not os.path.exists(task_path):
                task_path = "face_landmarker.task"
            if not os.path.exists(task_path):
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, task_path)

            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            _detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            log.warning("MediaPipe FaceLandmarker initialization error: %s", e)
            _detector = False
    return _detector if _detector is not False else None


def _extract_mouth_crop(frame_bgr: np.ndarray, img_size: int = 96) -> np.ndarray:
    """Extracts exact mouth crop using 3D lip landmarks, with padding."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    det = _get_detector()

    # 1. MediaPipe 3D Landmark Detection
    if det is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(mp_img)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in MOUTH_IDXS])
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                mw = x2 - x1
                mh = y2 - y1
                # Add 35% padding around the lips
                pad = int(max(mw, mh) * 0.35)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                if x2 > x1 and y2 > y1:
                    crop = gray[y1:y2, x1:x2]
                    return cv2.resize(crop, (img_size, img_size))
        except Exception:
            pass

    # 2. Geometric fallback
    my1 = int(h * 0.28)
    my2 = int(h * 0.58)
    mx1 = int(w * 0.28)
    mx2 = int(w * 0.72)
    crop = gray[my1:my2, mx1:mx2]
    return cv2.resize(crop, (img_size, img_size))


def load_lip_frames(
    video_path: str,
    n_frames: int,
    img_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        crops.append(_extract_mouth_crop(frame, img_size))
    cap.release()

    if not crops:
        return None, None

    while len(crops) < n_frames:
        crops.append(crops[-1])
        raws.append(raws[-1])

    crops = crops[:n_frames]
    raws  = raws[:n_frames]

    crops_arr = np.stack(crops, axis=0)
    raws_arr  = np.stack(raws, axis=0)
    crops_f   = crops_arr[:, np.newaxis, :, :].astype(np.float32) / 255.0
    return crops_f, raws_arr


def load_audio_16k(
    video_path: str,
    audio_len: int,
    sr: int = 16_000,
) -> Optional[np.ndarray]:
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
            return None
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None

    if len(wav) == 0:
        return None
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
    frames, raws = load_lip_frames(video_path, n_frames, img_size)
    audio = load_audio_16k(video_path, audio_len, sr)
    ok = (frames is not None) and (audio is not None)
    err = ("No video frames. " if frames is None else "") + ("No audio track." if audio is None else "")
    return dict(frames=frames, audio=audio, raw_frames=raws, ok=ok, error=err.strip())
