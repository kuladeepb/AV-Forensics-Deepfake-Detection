"""
main.py  –  AV-Forensics Backend FastAPI application
"""

from __future__ import annotations
import hashlib, logging, os, shutil, sys, tempfile, time, uuid
from pathlib import Path
from typing import Any, Dict

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_DIR = _BACKEND_DIR.parent
_ML_DIR = _REPO_DIR / "ml"

for p in [str(_BACKEND_DIR), str(_ML_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from backend_config import (
        ALLOWED_ORIGINS, APP_VERSION, CACHE_TTL_SEC,
        MAX_DURATION_SEC, MAX_UPLOAD_MB, MOCK_MODEL,
    )
except ImportError:
    try:
        from backend.backend_config import (
            ALLOWED_ORIGINS, APP_VERSION, CACHE_TTL_SEC,
            MAX_DURATION_SEC, MAX_UPLOAD_MB, MOCK_MODEL,
        )
    except ImportError:
        from config import (
            ALLOWED_ORIGINS, APP_VERSION, CACHE_TTL_SEC,
            MAX_DURATION_SEC, MAX_UPLOAD_MB, MOCK_MODEL,
        )

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="AV-Forensics API",
    description="Audio-Visual Lip-Sync Deepfake Detection",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory result cache ─────────────────────────────────────────────────────
_cache: Dict[str, dict] = {}   # case_id → {result, report_path, expires_at}


def _evict_expired():
    now = time.time()
    expired = [k for k, v in _cache.items() if v["expires_at"] < now]
    for k in expired:
        try:
            rp = _cache[k].get("report_path")
            if rp and os.path.exists(rp):
                os.remove(rp)
        except Exception:
            pass
        del _cache[k]


# ── Allowed MIME types ─────────────────────────────────────────────────────────
ALLOWED_MIME   = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"}
ALLOWED_EXT    = {".mp4", ".mov", ".avi"}
MAX_BYTES      = MAX_UPLOAD_MB * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "mock_mode": MOCK_MODEL,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    _evict_expired()

    # ── Validation ──────────────────────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}",
        )

    # Read into memory for size check, then write to temp file
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) / 1024 / 1024:.1f} MB). Max {MAX_UPLOAD_MB} MB.",
        )

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    tmp_path = tmp.name

    try:
        # Duration check (via ffprobe) — skip in mock mode for speed
        if not MOCK_MODEL:
            import subprocess
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            try:
                dur = float(proc.stdout.strip())
                if dur > MAX_DURATION_SEC:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Video too long ({dur:.1f}s). Max {MAX_DURATION_SEC}s.",
                    )
            except ValueError:
                pass   # ffprobe failed → continue

        # ── Inference (run in worker thread so event loop is not blocked) ─────
        import asyncio
        from inference import analyze_video
        result = await asyncio.to_thread(analyze_video, tmp_path, file.filename or "upload")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        # Always delete the uploaded video
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # ── Cache & respond ────────────────────────────────────────────────────
    case_id = str(uuid.uuid4())
    _cache[case_id] = {
        "result":      result,
        "report_path": None,
        "expires_at":  time.time() + CACHE_TTL_SEC,
    }

    return JSONResponse({"case_id": case_id, **result})


@app.get("/api/report/{case_id}")
def get_report(case_id: str):
    _evict_expired()
    entry = _cache.get(case_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found or expired. Results are cached for ~1 hour.",
        )

    # Generate PDF lazily
    if entry["report_path"] is None or not os.path.exists(entry.get("report_path", "")):
        from report import generate_report
        pdf_dir  = Path(tempfile.gettempdir()) / "avforensics_reports"
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = str(pdf_dir / f"report_{case_id}.pdf")
        try:
            generate_report(entry["result"], case_id, pdf_path)
            entry["report_path"] = pdf_path
        except Exception as e:
            log.exception("Report generation error")
            raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    fname = f"forensic_report_{case_id[:8]}.pdf"
    return FileResponse(
        entry["report_path"],
        media_type="application/pdf",
        filename=fname,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/result/{case_id}")
def get_result(case_id: str):
    """Retrieve cached result JSON (useful if frontend needs to re-fetch)."""
    _evict_expired()
    entry = _cache.get(case_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found or expired.",
        )
    return JSONResponse({"case_id": case_id, **entry["result"]})

