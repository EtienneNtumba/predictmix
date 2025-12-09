from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_pr(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """
    Plot ROC and Precision–Recall curves from a CSV containing
    'y_true' and 'risk_proba'.
    """
    results_path = Path(results_path)
    df = pd.read_csv(results_path)

    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")

    y_true = df["y_true"].values
    y_score = df["risk_proba"].values

    out_dir = _ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("PredictMix – ROC curve")
    plt.legend(loc="lower right")
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["roc"] = str(roc_path)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PredictMix – Precision–Recall curve")
    plt.legend(loc="lower left")
    pr_path = out_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["pr"] = str(pr_path)

    return paths


def plot_histograms(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """
    Plot histograms of predicted risk, overall and stratified by class.
    """
    results_path = Path(results_path)
    df = pd.read_csv(results_path)

    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")

    y_true = df["y_true"].values
    y_score = df["risk_proba"].values

    out_dir = _ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    # Overall
    plt.figure()
    plt.hist(y_score, bins=20)
    plt.xlabel("Predicted risk (risk_proba)")
    plt.ylabel("Count")
    plt.title("PredictMix – risk distribution (all samples)")
    hist_all_path = out_dir / "hist_risk_all.png"
    plt.savefig(hist_all_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["hist_all"] = str(hist_all_path)

    # By class
    plt.figure()
    plt.hist(y_score[y_true == 0], bins=20, alpha=0.7, label="Class 0")
    plt.hist(y_score[y_true == 1], bins=20, alpha=0.7, label="Class 1")
    plt.xlabel("Predicted risk (risk_proba)")
    plt.ylabel("Count")
    plt.title("PredictMix – risk distribution by class")
    plt.legend()
    hist_by_class_path = out_dir / "hist_risk_by_class.png"
    plt.savefig(hist_by_class_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["hist_by_class"] = str(hist_by_class_path)

    return paths


def plot_scatter(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """
    Scatter plot of predicted risk vs. true class (with jitter),
    useful to visualize class separation.
    """
    results_path = Path(results_path)
    df = pd.read_csv(results_path)

    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")

    y_true = df["y_true"].values
    y_score = df["risk_proba"].values

    out_dir = _ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    # Jitter on y-axis (class labels)
    y_jitter = y_true + (np.random.rand(len(y_true)) - 0.5) * 0.1

    plt.figure()
    plt.scatter(y_score, y_jitter, s=20)
    plt.yticks([0, 1], ["Class 0", "Class 1"])
    plt.xlabel("Predicted risk (risk_proba)")
    plt.ylabel("True class (jittered)")
    plt.title("PredictMix – scatter of risk by class")
    scatter_path = out_dir / "scatter_risk_vs_class.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["scatter"] = str(scatter_path)

    return paths


def plot_confusion_heatmap(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    threshold: float = 0.5,
) -> Dict[str, str]:
    """
    Confusion matrix heatmap using a probability threshold (default 0.5).
    """
    results_path = Path(results_path)
    df = pd.read_csv(results_path)

    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")

    y_true = df["y_true"].values
    y_score = df["risk_proba"].values
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    out_dir = _ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"PredictMix – confusion matrix (threshold={threshold:.2f})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    heatmap_path = out_dir / "confusion_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["confusion_heatmap"] = str(heatmap_path)

    return paths


def plot_calibration(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    n_bins: int = 10,
) -> Dict[str, str]:
    """
    Reliability diagram (calibration curve).
    """
    results_path = Path(results_path)
    df = pd.read_csv(results_path)

    if "y_true" not in df.columns or "risk_proba" not in df.columns:
        raise ValueError("results CSV must contain 'y_true' and 'risk_proba' columns.")

    y_true = df["y_true"].values
    y_score = df["risk_proba"].values

    out_dir = _ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins)

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("PredictMix – calibration curve")
    plt.legend(loc="upper left")
    calib_path = out_dir / "calibration_curve.png"
    plt.savefig(calib_path, dpi=300, bbox_inches="tight")
    plt.close()
    paths["calibration"] = str(calib_path)

    return paths


def plot_all_from_results(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
    kind: str = "all",
) -> Dict[str, str]:
    """
    High-level helper to generate multiple plots from a results CSV.

    kind can be:
      - 'rocpr'   : ROC + PR curves
      - 'hist'    : histograms of predicted risk
      - 'scatter' : scatter risk vs class
      - 'heatmap' : confusion matrix heatmap
      - 'calib'   : calibration (reliability) curve
      - 'all'     : all of the above
    """
    kinds = {"rocpr", "hist", "scatter", "heatmap", "calib", "all"}
    if kind not in kinds:
        raise ValueError(f"Unknown kind '{kind}'. Must be one of: {sorted(kinds)}")

    paths: Dict[str, str] = {}
    if kind in ("rocpr", "all"):
        paths.update(plot_roc_pr(results_path, output_dir))
    if kind in ("hist", "all"):
        paths.update(plot_histograms(results_path, output_dir))
    if kind in ("scatter", "all"):
        paths.update(plot_scatter(results_path, output_dir))
    if kind in ("heatmap", "all"):
        paths.update(plot_confusion_heatmap(results_path, output_dir))
    if kind in ("calib", "all"):
        paths.update(plot_calibration(results_path, output_dir))

    return paths


# Backward-compatibility: previously we exposed plot_curves()
def plot_curves(
    results_path: str | Path,
    output_dir: str | Path = "predictmix_plots",
) -> Dict[str, str]:
    """
    Backward-compatible wrapper: generates ROC and PR curves only.
    """
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
    Volcano plot for GWAS-like summary statistics.

    The input CSV must contain:
      - effect_col: effect size (e.g., beta, log(OR))
      - pval_col:   p-value

    X-axis: effect size
    Y-axis: -log10(p-value)
    """
    summary_path = Path(summary_path)
    df = pd.read_csv(summary_path)

    if effect_col not in df.columns or pval_col not in df.columns:
        raise ValueError(
            f"Summary stats must contain '{effect_col}' (effect) and "
            f"'{pval_col}' (p-value) columns."
        )

    effect = df[effect_col].values
    pvals = df[pval_col].values
    minus_log10_p = -np.log10(pvals)

    plt.figure()
    plt.scatter(effect, minus_log10_p, s=10)

    # Threshold lines
    gw_y = -np.log10(genome_wide_threshold)
    sug_y = -np.log10(suggestive_threshold)
    plt.axhline(gw_y, linestyle="--")
    plt.axhline(sug_y, linestyle=":")

    plt.xlabel(effect_col)
    plt.ylabel(r"-log10(p-value)")
    plt.title("PredictMix – volcano plot")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)

