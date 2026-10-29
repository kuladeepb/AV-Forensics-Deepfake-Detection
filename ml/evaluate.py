"""
evaluate.py  –  AV-Forensics
Full evaluation: Task 1 (Real/Fake) and Task 2 (4-class modality).
Plots confusion matrices, ROC curve, loss/accuracy curves.
No hard-coded numbers — all metrics are computed from model predictions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from config import CLASSES


# ── Collect predictions ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    bin_true, bin_pred, bin_prob = [], [], []
    cls_true, cls_pred            = [], []
    afake_true, afake_pred        = [], []
    vfake_true, vfake_pred        = [], []

    for batch in loader:
        frames, audio, y_bin, y_4, y_afake, y_vfake = [b.to(device) for b in batch]
        out = model(frames, audio)

        p_bin = torch.softmax(out["logits_bin"], dim=1)[:, 1].cpu().numpy()
        bin_prob.extend(p_bin.tolist())
        bin_pred.extend(out["logits_bin"].argmax(1).cpu().tolist())
        bin_true.extend(y_bin.cpu().tolist())

        cls_pred.extend(out["logits_4"].argmax(1).cpu().tolist())
        cls_true.extend(y_4.cpu().tolist())

        afake_pred.extend(out["logits_afake"].argmax(1).cpu().tolist())
        afake_true.extend(y_afake.cpu().tolist())

        vfake_pred.extend(out["logits_vfake"].argmax(1).cpu().tolist())
        vfake_true.extend(y_vfake.cpu().tolist())

    return {
        "bin_true":   np.array(bin_true),
        "bin_pred":   np.array(bin_pred),
        "bin_prob":   np.array(bin_prob),
        "cls_true":   np.array(cls_true),
        "cls_pred":   np.array(cls_pred),
        "afake_true": np.array(afake_true),
        "afake_pred": np.array(afake_pred),
        "vfake_true": np.array(vfake_true),
        "vfake_pred": np.array(vfake_pred),
    }


# ── Task 1 metrics ─────────────────────────────────────────────────────────────

def task1_metrics(preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    bt, bp, prob = preds["bin_true"], preds["bin_pred"], preds["bin_prob"]
    p, r, f1, _ = precision_recall_fscore_support(bt, bp, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(bt, prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":  accuracy_score(bt, bp),
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "roc_auc":   auc,
    }


def plot_task1(preds: Dict[str, np.ndarray], save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = confusion_matrix(preds["bin_true"], preds["bin_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
        ax=axes[0],
    )
    axes[0].set_title("Task 1 – Real/Fake Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(preds["bin_true"], preds["bin_prob"])
        auc = roc_auc_score(preds["bin_true"], preds["bin_prob"])
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_title("Task 1 – ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()
    except ValueError as e:
        axes[1].text(0.5, 0.5, f"ROC unavailable:\n{e}", ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Task 2 metrics ─────────────────────────────────────────────────────────────

def task2_metrics(preds: Dict[str, np.ndarray]) -> Dict:
    ct, cp = preds["cls_true"], preds["cls_pred"]
    report = classification_report(ct, cp, target_names=CLASSES, zero_division=0)
    macro_f1 = f1_score(ct, cp, average="macro", zero_division=0)
    acc = accuracy_score(ct, cp)
    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}


def plot_task2(preds: Dict[str, np.ndarray], save_path: Optional[str] = None):
    cm = confusion_matrix(preds["cls_true"], preds["cls_pred"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
    )
    plt.title("Task 2 – 4-Class Modality Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Training curves ────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    train_loss = [h["loss"] for h in history["train"]]
    val_loss   = [h["loss"] for h in history["val"]]
    val_bin_f1 = [h["bin_f1"] for h in history["val"]]
    val_4_f1   = [h["macro_f1_4"] for h in history["val"]]

    epochs = range(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss,   label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_bin_f1, label="val_bin_macro_f1", color="green")
    axes[1].set_title("Task 1 – Validation Binary F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, val_4_f1, label="val_4class_macro_f1", color="orange")
    axes[2].set_title("Task 2 – Validation 4-Class Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Full evaluation report ─────────────────────────────────────────────────────

def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    history: Optional[Dict] = None,
    plot_dir: Optional[str] = None,
):
    print("=" * 60)
    print("FULL EVALUATION REPORT")
    print("=" * 60)

    preds = collect_predictions(model, loader, device)

    # Task 1
    t1 = task1_metrics(preds)
    print("\n── Task 1: Real/Fake Detection ──")
    print(f"  Accuracy : {t1['accuracy']:.4f}")
    print(f"  Precision: {t1['precision']:.4f}")
    print(f"  Recall   : {t1['recall']:.4f}")
    print(f"  F1       : {t1['f1']:.4f}")
    print(f"  ROC-AUC  : {t1['roc_auc']:.4f}")
    plot_task1(preds, save_path=f"{plot_dir}/task1_plots.png" if plot_dir else None)

    # Task 2
    t2 = task2_metrics(preds)
    print("\n── Task 2: 4-Class Modality Classification ──")
    print(f"  Accuracy : {t2['accuracy']:.4f}")
    print(f"  Macro-F1 : {t2['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(t2["report"])
    plot_task2(preds, save_path=f"{plot_dir}/task2_cm.png" if plot_dir else None)

    # Training curves
    if history:
        plot_training_curves(
            history,
            save_path=f"{plot_dir}/training_curves.png" if plot_dir else None,
        )

    print("=" * 60)
    return {"task1": t1, "task2": t2}

