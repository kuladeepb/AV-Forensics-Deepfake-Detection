"""
backend_config.py  –  AV-Forensics Backend
All backend server constants in one place (named backend_config to prevent
module name collision with ml/config.py when run from repo root).
"""
import os

# ── Server ─────────────────────────────────────────────────────────────────────
MOCK_MODEL        = os.getenv("MOCK_MODEL", "1") == "1"
ALLOWED_ORIGINS   = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_UPLOAD_MB     = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_DURATION_SEC  = int(os.getenv("MAX_DURATION_SEC", "30"))
CACHE_TTL_SEC     = int(os.getenv("CACHE_TTL_SEC", "3600"))   # 1 h
APP_VERSION       = "1.0.0"

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "model.pth")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

# ── Evidence extraction ────────────────────────────────────────────────────────
TOP_EVIDENCE_N   = int(os.getenv("TOP_EVIDENCE_N", "6"))
MIN_FRAME_GAP    = int(os.getenv("MIN_FRAME_GAP",  "8"))

# ── Sliding-window inference ───────────────────────────────────────────────────
WINDOW_SEC = 4.0
STRIDE_SEC = 2.0
