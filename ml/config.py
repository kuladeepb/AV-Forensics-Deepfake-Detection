"""
Central configuration for AV-Forensics.
Swap AUDIO_MODEL_NAME here to change the audio encoder globally.
"""

# ── Audio encoder ──────────────────────────────────────────────────────────────
AUDIO_MODEL_NAME = "utter-project/mHuBERT-147"
# Swap to: "facebook/wav2vec2-xls-r-300m" for multilingual alternative

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASSES       = ["RVRA", "RVFA", "FVRA", "FVFA"]
CLASS_TO_ID   = {c: i for i, c in enumerate(CLASSES)}
BINARY_LABEL  = {"RVRA": 0, "RVFA": 1, "FVRA": 1, "FVFA": 1}  # 0=Real, 1=Fake
# Derived audio/video fake flags per class:
AUDIO_FAKE    = {"RVRA": 0, "RVFA": 1, "FVRA": 0, "FVFA": 1}
VIDEO_FAKE    = {"RVRA": 0, "RVFA": 0, "FVRA": 1, "FVFA": 1}
MODALITY_LABELS = {
    "RVRA": "Authentic",
    "RVFA": "Audio Manipulated",
    "FVRA": "Video Manipulated",
    "FVFA": "Both Manipulated",
}

# ── Preprocessing ──────────────────────────────────────────────────────────────
SR          = 16_000          # audio sample rate
N_FRAMES    = 32              # frames per 4-second window
IMG_SIZE    = 96              # mouth crop: 96×96 grayscale
AUDIO_LEN   = SR * 4         # 4-second audio window (samples)
WINDOW_SEC  = 4.0
STRIDE_SEC  = 2.0

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED_DIM       = 256
NUM_VIS_HEADS   = 4
NUM_VIS_LAYERS  = 4
NUM_CROSS_HEADS = 4
NUM_CROSS_LAYERS= 2
DROPOUT         = 0.1

# ── Training ───────────────────────────────────────────────────────────────────
LAMBDA_SYNC = 0.3          # weight for sync loss
AUX_WEIGHT  = 0.5          # weight for auxiliary audio/video fake heads
LR          = 1e-4
WEIGHT_DECAY= 1e-2
EPOCHS      = 40
BATCH_SIZE  = 6
GRAD_ACCUM  = 4            # effective batch = BATCH_SIZE * GRAD_ACCUM
PATIENCE    = 8            # early stopping patience (epochs)
VAL_FRAC    = 0.12         # fraction of train identities for validation

# ── Export ─────────────────────────────────────────────────────────────────────
FAKE_THRESHOLD  = 0.5      # binary logit threshold
TOP_EVIDENCE_N  = 6        # number of evidence frames to surface
MIN_FRAME_GAP   = 8        # min frame gap between evidence frames

