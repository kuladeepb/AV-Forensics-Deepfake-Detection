"""
dataset.py  –  AV-Forensics
PyTorch Dataset, identity-grouped split logic, and class-weight computation.
"""

from __future__ import annotations
import glob, logging, os, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    CLASSES, CLASS_TO_ID, BINARY_LABEL, AUDIO_FAKE, VIDEO_FAKE,
    SR, N_FRAMES, IMG_SIZE, AUDIO_LEN, VAL_FRAC, AUDIO_MODEL_NAME,
)
from preprocessing import preprocess_video

log = logging.getLogger(__name__)


# ── Identity extraction ────────────────────────────────────────────────────────

def _identity_from_path(path: str) -> str:
    """
    FakeAVCeleb filenames look like id00012_id00034_00001.mp4.
    We use the first id token as the source-speaker identity.
    Falls back to the filename stem if no match.
    """
    name = Path(path).stem
    m = re.match(r"(id\d+)", name)
    return m.group(1) if m else name


# ── File scanning ──────────────────────────────────────────────────────────────

def scan_split(data_root: str, split: str) -> List[Tuple[str, str]]:
    """
    Scan DATA_ROOT/<class>/<split>/*.mp4 and return (path, class_name) pairs.
    """
    items: List[Tuple[str, str]] = []
    for cls in CLASSES:
        pattern = os.path.join(data_root, cls, split, "*.mp4")
        paths = sorted(glob.glob(pattern))
        for p in paths:
            items.append((p, cls))
    return items


def identity_grouped_split(
    train_items: List[Tuple[str, str]],
    val_frac: float = VAL_FRAC,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split training items into train/val ensuring no identity appears in both.
    """
    rng = np.random.default_rng(seed)

    # Collect unique identities
    identities: List[str] = []
    seen = set()
    for path, _ in train_items:
        ident = _identity_from_path(path)
        if ident not in seen:
            identities.append(ident)
            seen.add(ident)

    identities = sorted(identities)
    rng.shuffle(identities)
    n_val = max(1, int(len(identities) * val_frac))
    val_ids = set(identities[:n_val])

    train_out, val_out = [], []
    for item in train_items:
        ident = _identity_from_path(item[0])
        (val_out if ident in val_ids else train_out).append(item)

    log.info(
        "Split: %d train / %d val   (%d val identities out of %d)",
        len(train_out), len(val_out), len(val_ids), len(identities),
    )
    return train_out, val_out


def compute_class_weights(items: List[Tuple[str, str]]) -> torch.Tensor:
    """Inverse-frequency class weights for 4-class head."""
    counts = np.zeros(len(CLASSES), dtype=float)
    for _, cls in items:
        counts[CLASS_TO_ID[cls]] += 1
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASSES)
    return torch.tensor(weights, dtype=torch.float32)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class AVDataset(Dataset):
    """
    Item: (frames, audio_input, binary_label, class_id, audio_fake, video_fake)
    frames      : Tensor (T, 1, H, W)
    audio_input : Tensor (audio_len,)  — raw waveform; feature extraction done in model
    binary_label: int  0=Real 1=Fake
    class_id    : int  0..3
    audio_fake  : int  0/1
    video_fake  : int  0/1
    """

    def __init__(
        self,
        items: List[Tuple[str, str]],
        cache_dir: Optional[str] = None,
        skip_bad: bool = True,
    ):
        self.items     = items
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.skip_bad  = skip_bad
        self._valid_indices = self._build_index()

    def _cache_path(self, video_path: str, suffix: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        stem = Path(video_path).stem
        parent = Path(video_path).parent.name  # e.g. "train"
        cls    = Path(video_path).parent.parent.name  # e.g. "RVRA"
        d = self.cache_dir / cls / parent
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stem}{suffix}"

    def _load_or_cache(self, video_path: str, cls_name: str) -> Optional[dict]:
        cp_f = self._cache_path(video_path, "_frames.npy")
        cp_a = self._cache_path(video_path, "_audio.npy")

        if cp_f and cp_a and cp_f.exists() and cp_a.exists():
            frames = np.load(str(cp_f))
            audio  = np.load(str(cp_a))
            return dict(frames=frames, audio=audio, ok=True, error="")

        result = preprocess_video(video_path, N_FRAMES, IMG_SIZE, AUDIO_LEN, SR)
        if result["ok"]:
            if cp_f and cp_a:
                np.save(str(cp_f), result["frames"])
                np.save(str(cp_a), result["audio"])
        return result

    def _build_index(self) -> List[int]:
        """Return valid item indices (skip files with no face/audio)."""
        valid = []
        for i, (path, _) in enumerate(self.items):
            # Quick check: just verify file exists; real validation on __getitem__
            if os.path.exists(path):
                valid.append(i)
            else:
                log.warning("Missing file: %s", path)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]
        video_path, cls_name = self.items[real_idx]

        result = self._load_or_cache(video_path, cls_name)
        if not result["ok"]:
            log.warning("Bad sample %s: %s — using zeros", video_path, result["error"])
            frames = np.zeros((N_FRAMES, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            audio  = np.zeros(AUDIO_LEN, dtype=np.float32)
        else:
            frames = result["frames"]   # (T, 1, H, W)
            audio  = result["audio"]    # (audio_len,)

        return (
            torch.from_numpy(frames),
            torch.from_numpy(audio),
            BINARY_LABEL[cls_name],
            CLASS_TO_ID[cls_name],
            AUDIO_FAKE[cls_name],
            VIDEO_FAKE[cls_name],
        )

