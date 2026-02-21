"""
plots.py
========
Publication-grade visualisation suite for PredictMix.

Functions
---------
plot_roc_pr             ROC curve + Precision-Recall curve
plot_histograms         Risk score distributions (overall + by class)
plot_scatter            Predicted risk vs. true class (jittered)
plot_confusion_heatmap  Confusion matrix heatmap
plot_calibration        Reliability diagram (calibration curve)
plot_all_from_results   High-level dispatcher (kind='all' generates all 7 plots)
plot_volcano            Volcano plot for GWAS summary statistics
plot_curves             Backward-compatible alias for plot_roc_pr
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_results(results_path: str | Path) -> tuple:
    df = pd.read_csv(Path(results_path))
    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")
    return df["y_true"].values, df["risk_proba"].values


# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """ROC curve + Precision-Recall curve."""
    y_true, y_score = _load_results(results_path)
    out = _ensure_dir(output_dir)
    paths = {}

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PredictMix — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    p = out / "roc_curve.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    paths["roc"] = str(p)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PredictMix — Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    p = out / "pr_curve.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    paths["pr"] = str(p)

    return paths


def plot_histograms(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """Risk score histograms — overall and stratified by true class."""
    y_true, y_score = _load_results(results_path)
    out = _ensure_dir(output_dir)
    paths = {}

    plt.figure(figsize=(6, 4))
    plt.hist(y_score, bins=25, edgecolor="black")
    plt.xlabel("Predicted risk probability")
    plt.ylabel("Count")
    plt.title("PredictMix — Risk Score Distribution (all samples)")
    plt.tight_layout()
    p = out / "hist_risk_all.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    paths["hist_all"] = str(p)

    plt.figure(figsize=(6, 4))
    plt.hist(y_score[y_true == 0], bins=20, alpha=0.7, label="Controls (y=0)", color="steelblue")
    plt.hist(y_score[y_true == 1], bins=20, alpha=0.7, label="Cases (y=1)", color="tomato")
    plt.xlabel("Predicted risk probability")
    plt.ylabel("Count")
    plt.title("PredictMix — Risk Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    p = out / "hist_risk_by_class.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    paths["hist_by_class"] = str(p)

    return paths


def plot_scatter(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """Scatter: predicted risk vs. true class (jittered)."""
    y_true, y_score = _load_results(results_path)
    out = _ensure_dir(output_dir)
    rng = np.random.default_rng(42)
    jitter = y_true + (rng.random(len(y_true)) - 0.5) * 0.12
    plt.figure(figsize=(6, 4))
    plt.scatter(y_score, jitter, s=12, alpha=0.6)
    plt.yticks([0, 1], ["Control (0)", "Case (1)"])
    plt.xlabel("Predicted risk probability")
    plt.ylabel("True class (jittered)")
    plt.title("PredictMix — Predicted Risk vs. True Class")
    plt.tight_layout()
    p = out / "scatter_risk_vs_class.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return {"scatter": str(p)}


def plot_confusion_heatmap(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    threshold: float = 0.5,
) -> Dict[str, str]:
    """Confusion matrix heatmap at given probability threshold."""
    y_true, y_score = _load_results(results_path)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    out = _ensure_dir(output_dir)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"PredictMix — Confusion Matrix (threshold={threshold:.2f})")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = out / "confusion_heatmap.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return {"confusion_heatmap": str(p)}


def plot_calibration(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    n_bins: int = 10,
) -> Dict[str, str]:
    """Reliability diagram — calibration of predicted probabilities."""
    y_true, y_score = _load_results(results_path)
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    out = _ensure_dir(output_dir)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("PredictMix — Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    p = out / "calibration_curve.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return {"calibration": str(p)}


def plot_all_from_results(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    kind: str = "all",
) -> Dict[str, str]:
    """
    High-level dispatcher.

    kind : 'rocpr' | 'hist' | 'scatter' | 'heatmap' | 'calib' | 'all'
    """
    valid = {"rocpr", "hist", "scatter", "heatmap", "calib", "all"}
    if kind not in valid:
        raise ValueError(f"Unknown kind '{kind}'. Must be one of {sorted(valid)}")
    paths: Dict[str, str] = {}
    if kind in ("rocpr",   "all"): paths.update(plot_roc_pr(results_path, output_dir))
    if kind in ("hist",    "all"): paths.update(plot_histograms(results_path, output_dir))
    if kind in ("scatter", "all"): paths.update(plot_scatter(results_path, output_dir))
    if kind in ("heatmap", "all"): paths.update(plot_confusion_heatmap(results_path, output_dir))
    if kind in ("calib",   "all"): paths.update(plot_calibration(results_path, output_dir))
    return paths


def plot_curves(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """Backward-compatible alias — generates ROC and PR curves only."""
    return plot_roc_pr(results_path, output_dir)


def plot_volcano(
    summary_path: str | Path,
    output_path: str | Path = "predictmix_volcano.png",
    effect_col: str = "beta",
    pval_col: str = "pval",
    genome_wide_threshold: float = 5e-8,
    suggestive_threshold: float = 1e-5,
) -> str:
    """
    Volcano plot for GWAS summary statistics.

    X-axis: effect size (beta / log-OR)
    Y-axis: -log10(p-value)
    """
    df = pd.read_csv(Path(summary_path))
    if effect_col not in df.columns or pval_col not in df.columns:
        raise ValueError(
            f"Summary stats must contain '{effect_col}' and '{pval_col}' columns."
        )
    effect = df[effect_col].values
    minus_log10_p = -np.log10(np.clip(df[pval_col].values, 1e-300, 1.0))
    plt.figure(figsize=(8, 5))
    plt.scatter(effect, minus_log10_p, s=8, alpha=0.6, c="steelblue")
    plt.axhline(-np.log10(genome_wide_threshold), linestyle="--", color="red",
                lw=1.2, label="Genome-wide (5×10⁻⁸)")
    plt.axhline(-np.log10(suggestive_threshold), linestyle=":",  color="orange",
                lw=1.0, label="Suggestive (1×10⁻⁵)")
    plt.xlabel(effect_col)
    plt.ylabel(r"$-\log_{10}$(p-value)")
    plt.title("PredictMix — Volcano Plot")
    plt.legend()
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)
