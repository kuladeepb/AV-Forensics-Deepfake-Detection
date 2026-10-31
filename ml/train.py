"""
train.py  –  AV-Forensics
Training loop for Google Colab (free T4).
Called from colab_train.ipynb after dataset and model setup.
"""

from __future__ import annotations
import json, logging, os, time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import (
    LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, GRAD_ACCUM,
    PATIENCE, LAMBDA_SYNC, AUX_WEIGHT,
)
from model import AVDeepfakeModel, compute_total_loss

log = logging.getLogger(__name__)


# ── Evaluation helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ce_bin: nn.CrossEntropyLoss,
    ce_4: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_bin_true, all_bin_pred = [], []
    all_4_true,   all_4_pred   = [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        frames, audio, y_bin, y_4, y_afake, y_vfake = [b.to(device) for b in batch]
        out = model(frames, audio)
        loss, _ = compute_total_loss(
            out, y_bin, y_4, y_afake, y_vfake,
            ce_bin, ce_4, LAMBDA_SYNC, AUX_WEIGHT,
        )
        total_loss += loss.item()
        n_batches  += 1

        all_bin_true.extend(y_bin.cpu().tolist())
        all_bin_pred.extend(out["logits_bin"].argmax(1).cpu().tolist())
        all_4_true.extend(y_4.cpu().tolist())
        all_4_pred.extend(out["logits_4"].argmax(1).cpu().tolist())

    macro_f1_bin = f1_score(all_bin_true, all_bin_pred, average="macro", zero_division=0)
    macro_f1_4   = f1_score(all_4_true,   all_4_pred,   average="macro", zero_division=0)
    bin_acc = np.mean(np.array(all_bin_true) == np.array(all_bin_pred))
    acc_4   = np.mean(np.array(all_4_true)   == np.array(all_4_pred))

    return {
        "loss":        total_loss / max(n_batches, 1),
        "bin_acc":     float(bin_acc),
        "bin_f1":      float(macro_f1_bin),
        "acc_4":       float(acc_4),
        "macro_f1_4":  float(macro_f1_4),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    model: AVDeepfakeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights_4: torch.Tensor,
    save_dir: str = "/content/drive/MyDrive/avforensics_checkpoints",
    use_amp: bool = True,
) -> Dict:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    ce_bin = nn.CrossEntropyLoss().to(device)
    ce_4   = nn.CrossEntropyLoss(weight=class_weights_4.to(device)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # Modern PyTorch AMP support (cleans up deprecation warnings in PyTorch 2.x+)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device.type == "cuda")
    except Exception:
        scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    def get_autocast():
        try:
            return torch.amp.autocast(device_type=device_type, enabled=use_amp and device.type == "cuda")
        except Exception:
            return autocast(enabled=use_amp and device.type == "cuda")

    best_f1    = -1.0
    patience_ctr = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            frames, audio, y_bin, y_4, y_afake, y_vfake = [b.to(device) for b in batch]

            with get_autocast():
                out = model(frames, audio)
                loss, breakdown = compute_total_loss(
                    out, y_bin, y_4, y_afake, y_vfake,
                    ce_bin, ce_4, LAMBDA_SYNC, AUX_WEIGHT,
                )
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += breakdown["total"]

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, ce_bin, ce_4, device)
        train_metrics = {"loss": epoch_loss / len(train_loader)}
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_f1 = val_metrics["macro_f1_4"]
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | "
            "val_bin_f1=%.4f | val_4_f1=%.4f | lr=%.2e | %.0fs",
            epoch, EPOCHS,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics["bin_f1"], val_f1, lr_now, elapsed,
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_bin_acc={val_metrics['bin_acc']:.4f} | "
            f"val_4acc={val_metrics['acc_4']:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        # Checkpoint every epoch
        ckpt_path = save_path / f"epoch_{epoch:02d}_f1_{val_f1:.4f}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }, ckpt_path)

        # Best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), save_path / "best_model.pth")
            log.info("  ↑ New best macro-F1: %.4f", best_f1)
            print(f"  ↑ New best macro-F1={best_f1:.4f} → saved best_model.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # Save history
    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

