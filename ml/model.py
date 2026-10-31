"""
model.py  –  AV-Forensics
Lip Video Transformer + mHuBERT-147 (or wav2vec2) + Cross-Modal Transformer.

Returns logits AND pre-fusion per-time-step embeddings for sync timeline.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModel, AutoFeatureExtractor
except ImportError:
    AutoModel = None
    AutoFeatureExtractor = None

from config import (
    AUDIO_MODEL_NAME, EMBED_DIM, NUM_VIS_HEADS, NUM_VIS_LAYERS,
    NUM_CROSS_HEADS, NUM_CROSS_LAYERS, DROPOUT, N_FRAMES, IMG_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Visual Encoder – Lip Video Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Split each grayscale mouth frame into patch tokens."""
    def __init__(self, img_size: int = IMG_SIZE, patch_size: int = 16, embed_dim: int = EMBED_DIM):
        super().__init__()
        n = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        B, T = x.shape[:2]
        x = x.flatten(0, 1)                          # (B*T, 1, H, W)
        x = self.proj(x)                              # (B*T, dim, ph, pw)
        x = x.flatten(2).transpose(1, 2)             # (B*T, n_patches, dim)
        x = x.unflatten(0, (B, T))                   # (B, T, n_patches, dim)
        return x


class LipVideoTransformer(nn.Module):
    """
    Encode a sequence of mouth-crop frames.
    Returns (B, T, embed_dim) — one token per frame.
    """
    def __init__(
        self,
        img_size: int  = IMG_SIZE,
        patch_size: int = 16,
        embed_dim: int  = EMBED_DIM,
        n_heads: int    = NUM_VIS_HEADS,
        n_layers: int   = NUM_VIS_LAYERS,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        n_p = self.patch_embed.n_patches

        # Spatial position
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_p, embed_dim) * 0.02)
        # Spatial transformer (per frame)
        sp_layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, embed_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.spatial_enc = nn.TransformerEncoder(sp_layer, num_layers=2)

        # Temporal position
        self.temporal_pos = nn.Parameter(torch.randn(1, N_FRAMES, embed_dim) * 0.02)
        # Temporal transformer
        tp_layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, embed_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.temporal_enc = nn.TransformerEncoder(tp_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        B, T = x.shape[:2]
        patches = self.patch_embed(x)                    # (B, T, n_p, dim)
        patches = patches + self.spatial_pos             # broadcast

        # Spatial attention per frame
        patches = patches.flatten(0, 1)                  # (B*T, n_p, dim)
        patches = self.spatial_enc(patches)              # (B*T, n_p, dim)
        frame_tokens = patches.mean(1)                   # (B*T, dim)
        frame_tokens = frame_tokens.unflatten(0, (B, T)) # (B, T, dim)

        # Temporal attention across frames
        frame_tokens = frame_tokens + self.temporal_pos[:, :T, :]
        out = self.temporal_enc(frame_tokens)            # (B, T, dim)
        out = self.norm(out)
        return self.out_proj(out)                        # (B, T, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Audio Encoder – mHuBERT-147 with learnable weighted sum of hidden layers
# ═══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """
    Wraps mHuBERT-147 (or wav2vec2).
    - Freezes the CNN feature extractor (always).
    - Optionally freezes the lower `freeze_layers` transformer layers.
    - Learnable weighted sum over all hidden states → project to embed_dim.
    """

    def __init__(
        self,
        model_name: str = AUDIO_MODEL_NAME,
        embed_dim: int  = EMBED_DIM,
        freeze_cnn: bool = True,
        freeze_layers: int = 6,        # freeze bottom 6 transformer layers
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("The 'transformers' package is required to load pretrained audio encoders. Run 'pip install transformers'.")
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        # Freeze CNN feature extractor
        if freeze_cnn:
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad_(False)

        # Freeze lower transformer layers
        if hasattr(self.backbone, "encoder"):
            for i, layer in enumerate(self.backbone.encoder.layers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad_(False)

        # Use the hidden layer count + 1 for embedding layer (13 in mHuBERT)
        n_layers = getattr(self.backbone.config, "num_hidden_layers", 12) + 1
        hidden   = self.backbone.config.hidden_size
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: (B, audio_len)
        Returns: (B, Ta, embed_dim)
        """
        out = self.backbone(input_values=input_values)
        # hidden_states: tuple of tensors each (B, Ta, hidden)
        hidden = torch.stack(out.hidden_states, dim=1)   # (B, K, Ta, hidden)
        K = hidden.shape[1]
        if self.layer_weights.shape[0] != K:
            # Handle variable hidden state count across HuggingFace model architectures
            if self.layer_weights.shape[0] >= K:
                w = F.softmax(self.layer_weights[:K], dim=0)
            else:
                w = F.softmax(F.pad(self.layer_weights, (0, K - self.layer_weights.shape[0])), dim=0)
        else:
            w = F.softmax(self.layer_weights, dim=0)
        fused = (hidden * w[None, :, None, None]).sum(1) # (B, Ta, hidden)
        return self.proj(fused)                          # (B, Ta, embed_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cross-Modal Transformer Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalFusion(nn.Module):
    """
    Concatenate visual tokens V (B, Tv, dim) and audio tokens A (B, Tv, dim),
    run a transformer encoder over the combined sequence, then mean-pool → (B, dim).
    """
    def __init__(
        self,
        embed_dim: int  = EMBED_DIM,
        n_heads: int    = NUM_CROSS_HEADS,
        n_layers: int   = NUM_CROSS_LAYERS,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        # Modality type embeddings
        self.vis_type  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.aud_type  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, embed_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, v: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        v, a: (B, T, dim) — must be same length (T).
        Returns:
          pooled    (B, dim)
          v_out     (B, T, dim) unimodal visual (pre-fusion)
          a_out     (B, T, dim) unimodal audio  (pre-fusion)
        """
        v_out = v
        a_out = a
        v = v + self.vis_type
        a = a + self.aud_type
        tokens = torch.cat([v, a], dim=1)       # (B, 2T, dim)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(1)                 # (B, dim)
        return pooled, v_out, a_out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class AVDeepfakeModel(nn.Module):
    """
    Audio-Visual Lip-Sync Deepfake Detector.

    Heads:
      head_bin   – Real/Fake (2-class)
      head_4     – Modality class (4-class: RVRA/RVFA/FVRA/FVFA)
      head_afake – auxiliary audio fake (2-class)
      head_vfake – auxiliary video fake (2-class)

    forward() returns a dict with all logits and pre-fusion embeddings.
    """

    def __init__(
        self,
        embed_dim: int    = EMBED_DIM,
        audio_model: str  = AUDIO_MODEL_NAME,
        freeze_cnn: bool  = True,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.visual_enc = LipVideoTransformer(embed_dim=embed_dim)
        self.audio_enc  = AudioEncoder(audio_model, embed_dim, freeze_cnn, freeze_layers)
        self.fusion     = CrossModalFusion(embed_dim)

        def _head(out: int) -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(embed_dim // 2, out),
            )

        self.head_bin   = _head(2)
        self.head_4     = _head(4)
        self.head_afake = _head(2)  # on audio-only pooled embedding
        self.head_vfake = _head(2)  # on video-only pooled embedding

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        frames : (B, T, 1, H, W)
        audio  : (B, audio_len)

        Returns dict:
          logits_bin   (B, 2)
          logits_4     (B, 4)
          logits_afake (B, 2)
          logits_vfake (B, 2)
          vis_emb      (B, Tv, dim)   pre-fusion visual tokens
          aud_emb      (B, Tv, dim)   pre-fusion audio tokens (aligned to Tv)
        """
        v = self.visual_enc(frames)           # (B, Tv, dim)
        a = self.audio_enc(audio)             # (B, Ta, dim)

        # Align audio length to visual length via linear interpolation
        Tv = v.size(1)
        a = F.interpolate(
            a.transpose(1, 2),               # (B, dim, Ta)
            size=Tv,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)                    # (B, Tv, dim)

        pooled, v_pre, a_pre = self.fusion(v, a)

        # Unimodal pooled embeddings for auxiliary heads
        v_pool = v_pre.mean(1)               # (B, dim)
        a_pool = a_pre.mean(1)               # (B, dim)

        return {
            "logits_bin":   self.head_bin(pooled),
            "logits_4":     self.head_4(pooled),
            "logits_afake": self.head_afake(a_pool),
            "logits_vfake": self.head_vfake(v_pool),
            "vis_emb":      v_pre,           # (B, Tv, dim) — for sync timeline
            "aud_emb":      a_pre,           # (B, Tv, dim) — for sync timeline
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def sync_loss_fn(
    vis_emb: torch.Tensor,   # (B, T, dim)
    aud_emb: torch.Tensor,   # (B, T, dim)
    real_mask: torch.Tensor, # (B,) bool — True for RVRA samples
) -> torch.Tensor:
    """
    Sync loss applied ONLY to real (RVRA) pairs.
    - Cosine-similarity maximisation on aligned steps.
    - Contrastive term: push time-shifted audio as negatives.
    Returns scalar loss (0 if no real samples in batch).
    """
    if real_mask.sum() == 0:
        return vis_emb.new_tensor(0.0)

    v = F.normalize(vis_emb[real_mask], dim=-1)   # (n_real, T, dim)
    a = F.normalize(aud_emb[real_mask], dim=-1)   # (n_real, T, dim)

    # Positive: per-time-step cosine similarity → maximise → loss = 1 - sim
    pos_sim = (v * a).sum(-1).mean()              # scalar

    # Negative: time-shifted audio (shift by T//4)
    shift = max(1, v.size(1) // 4)
    a_neg = torch.roll(a, shifts=shift, dims=1)
    neg_sim = (v * a_neg).sum(-1).mean()

    # Contrastive: margin hinge
    margin = 0.3
    contrastive = F.relu(neg_sim - pos_sim + margin)

    return (1.0 - pos_sim) + contrastive


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    y_bin: torch.Tensor,
    y_4: torch.Tensor,
    y_afake: torch.Tensor,
    y_vfake: torch.Tensor,
    ce_bin: nn.CrossEntropyLoss,
    ce_4: nn.CrossEntropyLoss,
    lambda_sync: float = 0.3,
    aux_weight: float  = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss = CE_bin + CE_4 + aux*(CE_afake + CE_vfake) + lambda_sync * sync_loss
    sync_loss only on real (RVRA) samples.
    """
    ce_bin_loss  = ce_bin(outputs["logits_bin"],   y_bin)
    ce_4_loss    = ce_4(outputs["logits_4"],        y_4)
    ce_afake     = F.cross_entropy(outputs["logits_afake"], y_afake)
    ce_vfake     = F.cross_entropy(outputs["logits_vfake"], y_vfake)

    real_mask = (y_bin == 0)
    s_loss = sync_loss_fn(outputs["vis_emb"], outputs["aud_emb"], real_mask)

    total = (
        ce_bin_loss
        + ce_4_loss
        + aux_weight * (ce_afake + ce_vfake)
        + lambda_sync * s_loss
    )

    breakdown = {
        "ce_bin":  ce_bin_loss.item(),
        "ce_4":    ce_4_loss.item(),
        "ce_afake": ce_afake.item(),
        "ce_vfake": ce_vfake.item(),
        "sync":    s_loss.item(),
        "total":   total.item(),
    }
    return total, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Shape assertions (run with `python model.py`)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("Running shape-assertion tests (CPU, no real weights)…")

    B, T, H, W = 2, N_FRAMES, IMG_SIZE, IMG_SIZE
    frames = torch.randn(B, T, 1, H, W)
    audio  = torch.randn(B, 64_000)

    # Build model WITHOUT loading real weights
    model = AVDeepfakeModel.__new__(AVDeepfakeModel)
    nn.Module.__init__(model)
    model.visual_enc = LipVideoTransformer()
    model.audio_enc  = nn.Sequential(
        nn.Linear(1, 768),  # dummy — just tests shape logic
    )
    # Use a simpler audio stub for offline test
    class _DummyAudioEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64_000, EMBED_DIM)
        def forward(self, x):
            # x: (B, audio_len) → (B, 32, dim) stub
            B = x.shape[0]
            out = x[:, :N_FRAMES * EMBED_DIM].reshape(B, N_FRAMES, EMBED_DIM)
            return out

    model.audio_enc  = _DummyAudioEnc()
    model.fusion     = CrossModalFusion()

    def _head(out):
        return nn.Sequential(nn.LayerNorm(EMBED_DIM), nn.Linear(EMBED_DIM, out))

    model.head_bin   = _head(2)
    model.head_4     = _head(4)
    model.head_afake = _head(2)
    model.head_vfake = _head(2)

    # Patch forward to use dummy audio enc
    class _ModelStub(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.__dict__.update(m.__dict__)
        def forward(self, frames, audio):
            v = self.visual_enc(frames)
            a = self.audio_enc(audio)
            Tv = v.size(1)
            a = F.interpolate(a.transpose(1,2), size=Tv, mode='linear', align_corners=False).transpose(1,2)
            pooled, v_pre, a_pre = self.fusion(v, a)
            v_pool = v_pre.mean(1)
            a_pool = a_pre.mean(1)
            return {
                "logits_bin":   self.head_bin(pooled),
                "logits_4":     self.head_4(pooled),
                "logits_afake": self.head_afake(a_pool),
                "logits_vfake": self.head_vfake(v_pool),
                "vis_emb":      v_pre,
                "aud_emb":      a_pre,
            }

    stub = _ModelStub(model)
    stub.eval()
    with torch.no_grad():
        out = stub(frames, audio)

    assert out["logits_bin"].shape   == (B, 2),           f"logits_bin: {out['logits_bin'].shape}"
    assert out["logits_4"].shape     == (B, 4),           f"logits_4: {out['logits_4'].shape}"
    assert out["logits_afake"].shape == (B, 2),           f"logits_afake: {out['logits_afake'].shape}"
    assert out["logits_vfake"].shape == (B, 2),           f"logits_vfake: {out['logits_vfake'].shape}"
    assert out["vis_emb"].shape      == (B, N_FRAMES, EMBED_DIM), f"vis_emb: {out['vis_emb'].shape}"
    assert out["aud_emb"].shape      == (B, N_FRAMES, EMBED_DIM), f"aud_emb: {out['aud_emb'].shape}"

    # Test sync loss
    real_mask = torch.tensor([True, False])
    sl = sync_loss_fn(out["vis_emb"], out["aud_emb"], real_mask)
    assert sl.shape == (), f"sync_loss shape: {sl.shape}"

    print("[PASS] All shape assertions passed.")
    print(f"  logits_bin   : {out['logits_bin'].shape}")
    print(f"  logits_4     : {out['logits_4'].shape}")
    print(f"  logits_afake : {out['logits_afake'].shape}")
    print(f"  logits_vfake : {out['logits_vfake'].shape}")
    print(f"  vis_emb      : {out['vis_emb'].shape}")
    print(f"  aud_emb      : {out['aud_emb'].shape}")
    print(f"  sync_loss    : {sl.item():.4f}")

