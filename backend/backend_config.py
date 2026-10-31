"""
backend_config.py  –  AV-Forensics Backend
All backend server constants in one place (named backend_config to prevent
module name collision with ml/config.py when run from repo root).
"""
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# Automatically read backend/.env or root .env
for env_path in [_DIR / ".env", _DIR.parent / ".env"]:
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# Default MOCK_MODEL to 0 if model.pth exists, otherwise 1
has_real_model = (_DIR / "model.pth").exists()
default_mock = "0" if has_real_model else "1"

# ── Server ─────────────────────────────────────────────────────────────────────
MOCK_MODEL        = os.getenv("MOCK_MODEL", default_mock) == "1"
ALLOWED_ORIGINS   = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_UPLOAD_MB     = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_DURATION_SEC  = int(os.getenv("MAX_DURATION_SEC", "30"))
CACHE_TTL_SEC     = int(os.getenv("CACHE_TTL_SEC", "3600"))   # 1 h
APP_VERSION       = "1.0.0"

def _resolve_path(env_var: str, default_name: str) -> str:
    """
    Resolve a file path from an env var.
    Always falls back to an absolute path relative to this file's directory,
    so the server can be started from any working directory.
    """
    val = os.getenv(env_var, default_name)
    p = Path(val)
    # 1. Absolute / CWD-relative path that already exists
    if p.is_absolute() and p.exists():
        return str(p)
    # 2. Relative to backend/ directory (most common case)
    candidate = _DIR / val
    if candidate.exists():
        return str(candidate.resolve())
    # 3. Env var not set → use default name relative to backend/
    default_candidate = _DIR / default_name
    if default_candidate.exists():
        return str(default_candidate.resolve())
    # 4. Nothing found — still return absolute backend/ path so the error
    #    message is informative and not CWD-dependent.
    return str(_DIR / default_name)

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATH  = _resolve_path("MODEL_PATH",  "model.pth")
CONFIG_PATH = _resolve_path("CONFIG_PATH", "config.json")

# ── Evidence extraction ────────────────────────────────────────────────────────
TOP_EVIDENCE_N   = int(os.getenv("TOP_EVIDENCE_N", "6"))
MIN_FRAME_GAP    = int(os.getenv("MIN_FRAME_GAP",  "8"))

# ── Sliding-window inference ───────────────────────────────────────────────────
WINDOW_SEC = 4.0
STRIDE_SEC = 2.0

