"""
tests/test_api.py  –  AV-Forensics Backend
End-to-end tests using MOCK_MODEL=1 (no GPU, no real weights needed).
Run with:  MOCK_MODEL=1 pytest backend/tests/test_api.py -v
"""

import io, os, sys
os.environ["MOCK_MODEL"] = "1"   # must be set BEFORE importing main

import pytest
from fastapi.testclient import TestClient

from pathlib import Path

# Ensure backend and ml dirs are on path
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_ML_DIR = _BACKEND_DIR.parent / "ml"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_ML_DIR))

from main import app   # noqa: E402

client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["mock_mode"] is True


# ── /api/analyze — invalid inputs ─────────────────────────────────────────────

def test_analyze_wrong_extension():
    """Non-video file should return 415."""
    r = client.post(
        "/api/analyze",
        files={"file": ("evil.exe", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert r.status_code == 415


def test_analyze_too_large():
    """File over MAX_UPLOAD_MB should return 413."""
    big = b"0" * (51 * 1024 * 1024)   # 51 MB > 50 MB limit
    r = client.post(
        "/api/analyze",
        files={"file": ("big.mp4", big, "video/mp4")},
    )
    assert r.status_code == 413


# ── /api/analyze — mock mode ───────────────────────────────────────────────────

def test_analyze_mock_returns_valid_structure():
    """
    In mock mode, any valid .mp4 upload returns a well-formed response.
    Uses a minimal valid-header MP4 stub (4 bytes) — actual content
    doesn't matter in mock mode.
    """
    # Minimal ftyp box so it passes extension check
    fake_mp4 = b"\x00\x00\x00\x08ftyp"
    r = client.post(
        "/api/analyze",
        files={"file": ("test_video.mp4", fake_mp4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Required top-level keys
    for key in ["case_id", "verdict", "confidence", "modality",
                "sync_timeline", "suspect_segments", "evidence_frames"]:
        assert key in body, f"Missing key: {key}"

    # verdict
    assert body["verdict"] in ("REAL", "FAKE")

    # confidence: 0-1
    assert 0.0 <= body["confidence"] <= 1.0

    # modality
    mod = body["modality"]
    assert "class" in mod
    assert mod["class"] in ("RVRA", "RVFA", "FVRA", "FVFA")
    assert "label" in mod
    assert "probabilities" in mod
    assert "audio_fake_prob" in mod
    assert "video_fake_prob" in mod

    # sync_timeline is a list of {t, score}
    for entry in body["sync_timeline"]:
        assert "t" in entry
        assert "score" in entry
        assert 0.0 <= entry["score"] <= 1.0

    # evidence_frames
    for ef in body["evidence_frames"]:
        assert "t" in ef
        assert "score" in ef
        assert "frame_b64" in ef
        assert "mouth_b64" in ef

    assert bool(body["case_id"])


def test_get_result_after_analyze():
    """GET /api/result/{case_id} returns the cached result."""
    fake_mp4 = b"\x00\x00\x00\x08ftyp"
    r1 = client.post(
        "/api/analyze",
        files={"file": ("test.mp4", fake_mp4, "video/mp4")},
    )
    assert r1.status_code == 200
    case_id = r1.json()["case_id"]

    r2 = client.get(f"/api/result/{case_id}")
    assert r2.status_code == 200
    assert r2.json()["case_id"] == case_id


def test_get_result_unknown_case():
    """Unknown case_id returns 404."""
    r = client.get("/api/result/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_report_generation():
    """PDF report is generated and served."""
    fake_mp4 = b"\x00\x00\x00\x08ftyp"
    r1 = client.post(
        "/api/analyze",
        files={"file": ("sample.mp4", fake_mp4, "video/mp4")},
    )
    assert r1.status_code == 200
    case_id = r1.json()["case_id"]

    r2 = client.get(f"/api/report/{case_id}")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/pdf"
    # PDF starts with %PDF
    assert r2.content[:4] == b"%PDF"


def test_report_unknown_case():
    """PDF report for unknown case returns 404."""
    r = client.get("/api/report/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


# ── Preprocessing shape test (unit, no video needed) ──────────────────────────

def test_preprocessing_shapes():
    """Verify load_lip_frames / load_audio_16k return correct shapes on a dummy video."""
    import numpy as np
    import subprocess, tempfile, os

    # Create a minimal 2-second synthetic video with ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            "-t", "2", tmp_path,
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, timeout=30)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available for synthetic video test")

        # Add backend/../../ml to path
        ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml"))
        if ml_path not in sys.path:
            sys.path.insert(0, ml_path)

        from preprocessing import load_audio_16k
        SR, AUDIO_LEN = 16_000, 64_000
        wav = load_audio_16k(tmp_path, AUDIO_LEN, SR)
        assert wav is not None, "load_audio_16k returned None"
        assert wav.shape == (AUDIO_LEN,), f"Audio shape: {wav.shape}"
        assert wav.dtype == np.float32

    finally:
        os.remove(tmp_path)

