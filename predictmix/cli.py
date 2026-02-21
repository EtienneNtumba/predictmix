"""
cli.py
======
Command-line interface for PredictMix v0.2.0.

Commands
--------
  train          Train + evaluate a model on a CSV/Parquet dataset
  predict        Apply a saved model to new samples
  plot           Generate visualisation plots from a predictions CSV
  benchmark      Train and compare multiple models on the same dataset
  plot-volcano   Volcano plot from GWAS summary statistics
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .config import PredictMixConfig
from .pipeline import PredictMixPipeline
from .plots import plot_all_from_results, plot_volcano

warnings.filterwarnings("ignore")
app = typer.Typer(
    name="predictmix",
    help=(
        "PredictMix v0.2.0 — Integrated Polygenic + Clinical Disease Risk Prediction\n\n"
        "Models (18): logreg, bayesian, svm, rf, mlp, adaboost, bagging, ensemble,\n"
        "             coxph, deepsurv, gmm, kmeans_risk, transfer,\n"
        "             rnn, lstm, gru, cnn1d, transformer, sedl\n\n"
        "Feature selection (15): none, pearson, chi2, infogain, lasso, ridge,\n"
        "             elasticnet, tree, pca, rfe, dnn_l1, gated,\n"
        "             autoencoder, stability, stacked_dl"
    ),
    add_completion=False,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Risk category helper
# ─────────────────────────────────────────────────────────────────────────────

def _risk_category(p: float) -> str:
    if p > 0.80: return "Very High"
    if p > 0.60: return "High"
    if p > 0.40: return "Moderate"
    if p > 0.20: return "Low"
    return "Very Low"


# ─────────────────────────────────────────────────────────────────────────────
#  train
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to CSV/Parquet training dataset"),
    model: str = typer.Option(
        "ensemble", "--model", "-m",
        help=(
            "Model to train. Classical: logreg, bayesian, svm, rf, mlp, adaboost, "
            "bagging, ensemble. Survival: coxph, deepsurv. Stratification: gmm, "
            "kmeans_risk. Transfer: transfer. Deep: rnn, lstm, gru, cnn1d, "
            "transformer, sedl."
        ),
    ),
    feature_selection: str = typer.Option(
        "lasso", "--feature-selection", "-f",
        help=(
            "Feature selection method. Filter: none, pearson, chi2, infogain. "
            "Embedded: lasso, ridge, elasticnet, tree. Reduction: pca. "
            "Wrapper: rfe. Deep: dnn_l1, gated, autoencoder, stability, stacked_dl."
        ),
    ),
    n_features: int = typer.Option(100, "--n-features", "-k",
                                   help="Maximum number of features to keep."),
    target_column: str = typer.Option("y", "--target-column", "-y",
                                      help="Name of the binary target column (0/1)."),
    output_dir: str  = typer.Option("predictmix_output", "--output-dir", "-o",
                                    help="Directory for model and result files."),
    cv_folds: int    = typer.Option(5, "--cv-folds",
                                    help="Number of cross-validation folds."),
    test_size: float = typer.Option(0.20, "--test-size",
                                    help="Fraction of data held out for testing."),
    seed: int        = typer.Option(42, "--seed",
                                    help="Random seed for reproducibility."),
    export_predictions: Optional[str] = typer.Option(
        None, "--export-predictions",
        help="Optional CSV path to save per-sample predictions.",
    ),
    plots: bool      = typer.Option(False, "--plots/--no-plots",
                                    help="Auto-generate ROC + PR plots after training."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c",
                                              help="YAML config file (overrides all flags)."),
):
    """Train a PredictMix model and evaluate on a held-out test set."""
    # Load YAML config if provided
    if config_file:
        import yaml
        with open(config_file) as f:
            kw = yaml.safe_load(f)
        cfg = PredictMixConfig(**kw)
    else:
        cfg = PredictMixConfig(
            model=model,
            feature_selection=feature_selection,
            n_features=n_features,
            target_column=target_column,
            output_dir=output_dir,
            cv_folds=cv_folds,
            test_size=test_size,
            random_state=seed,
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export = export_predictions or str(out / "predictions.csv")

    typer.echo(f"\n🔬 PredictMix v0.2.0")
    typer.echo(f"   model            : {cfg.model}")
    typer.echo(f"   feature_selection: {cfg.feature_selection}")
    typer.echo(f"   n_features       : {cfg.n_features}")
    typer.echo(f"   target           : {cfg.target_column}")
    typer.echo(f"   output_dir       : {cfg.output_dir}\n")

    pipe = PredictMixPipeline(cfg)

    with typer.progressbar(length=1, label="Training…") as bar:
        metrics = pipe.fit(data_path, export_predictions=export)
        bar.update(1)

    typer.echo("\n📊 Cross-validation results:")
    for k, v in metrics["cv"].items():
        typer.echo(f"   {k:22s}: {v:.4f}")

    typer.echo("\n🏁 Test-set results:")
    for k, v in metrics["test"].items():
        typer.echo(f"   {k:22s}: {v:.4f}")

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    typer.echo(f"\n✅ Metrics saved to {metrics_path}")

    pipe.save()
    typer.echo(f"✅ Model saved to {out / 'predictmix_model.joblib'}")
    typer.echo(f"✅ Predictions saved to {export}")

    if plots:
        typer.echo("\n🎨 Generating plots…")
        paths = plot_all_from_results(export, out / "plots")
        for name, p in paths.items():
            typer.echo(f"   {name}: {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  predict
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def predict(
    model_path: str = typer.Argument(..., help="Path to saved .joblib model file"),
    data_path:  str = typer.Argument(..., help="Path to CSV with new samples"),
    output:     str = typer.Option("predictions.csv", "--output", "-o",
                                   help="Output CSV path"),
    threshold: float = typer.Option(0.5, "--threshold",
                                    help="Probability threshold for binary label"),
):
    """Apply a trained PredictMix model to new samples."""
    pipe = PredictMixPipeline.load(model_path)
    df   = pd.read_csv(data_path)
    proba = pipe.predict_proba(df)
    df["risk_proba"]    = proba
    df["risk_label"]    = (proba >= threshold).astype(int)
    df["risk_category"] = df["risk_proba"].apply(_risk_category)
    df.to_csv(output, index=False)
    typer.echo(f"✅ Predictions saved to {output}")
    typer.echo(f"   Samples scored: {len(df)}")
    typer.echo(f"   High-risk (≥{threshold}): {(proba >= threshold).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
#  plot
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def plot(
    results_path: str = typer.Argument(..., help="Path to predictions CSV"),
    kind: str = typer.Option(
        "all", "--kind",
        help="Plot type: rocpr | hist | scatter | heatmap | calib | all",
    ),
    output_dir: str = typer.Option("predictmix_plots", "--output-dir", "-o",
                                   help="Directory for plot files"),
):
    """Generate visualisation plots from a predictions CSV."""
    paths = plot_all_from_results(results_path, output_dir, kind=kind)
    typer.echo(f"✅ {len(paths)} plot(s) saved to {output_dir}")
    for name, p in paths.items():
        typer.echo(f"   {name}: {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  benchmark
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    data_path: str = typer.Argument(..., help="Path to CSV/Parquet training dataset"),
    models: str = typer.Option(
        "logreg,rf,ensemble,coxph,gmm,lstm,gru,cnn1d,transformer,deepsurv,sedl",
        "--models",
        help="Comma-separated list of model keys to benchmark.",
    ),
    feature_selection: str = typer.Option(
        "lasso", "--feature-selection", "-f",
        help="Feature selection method (applied to all models).",
    ),
    n_features: int = typer.Option(100, "--n-features", "-k"),
    target_column: str = typer.Option("y", "--target-column", "-y"),
    output_dir: str = typer.Option("benchmark_results", "--output-dir", "-o"),
    seed: int = typer.Option(42, "--seed"),
    cv_folds: int = typer.Option(5, "--cv-folds"),
):
    """
    Train and compare multiple models on the same dataset.

    Outputs a benchmark_summary.csv with AUC, Accuracy, F1, Precision, Recall.
    """
    import csv
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n🔬 PredictMix Benchmark — {len(model_list)} models")
    typer.echo(f"   feature_selection: {feature_selection}")
    typer.echo(f"   target           : {target_column}\n")

    rows = []
    for m in model_list:
        typer.echo(f"   Training {m}…", nl=False)
        cfg = PredictMixConfig(
            model=m,
            feature_selection=feature_selection,
            n_features=n_features,
            target_column=target_column,
            output_dir=str(out / m),
            cv_folds=cv_folds,
            random_state=seed,
        )
        try:
            pipe = PredictMixPipeline(cfg)
            metrics = pipe.fit(data_path)
            t = metrics["test"]
            rows.append({
                "model":     m,
                "AUC":       round(t["auc"], 4),
                "Accuracy":  round(t["accuracy"], 4),
                "F1":        round(t["f1_macro"], 4),
                "Precision": round(t["precision_macro"], 4),
                "Recall":    round(t["recall_macro"], 4),
            })
            typer.echo(f" AUC={t['auc']:.4f}")
        except Exception as e:
            typer.echo(f" FAILED — {e}")
            rows.append({"model": m, "AUC": "error", "Accuracy": "error",
                         "F1": "error", "Precision": "error", "Recall": "error"})

    summary_path = out / "benchmark_summary.csv"
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        typer.echo(f"\n📊 Benchmark summary:\n{df.to_string(index=False)}")
        typer.echo(f"\n✅ Summary saved to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  plot-volcano
# ─────────────────────────────────────────────────────────────────────────────

@app.command(name="plot-volcano")
def plot_volcano_cmd(
    summary_path: str = typer.Argument(..., help="Path to GWAS summary stats CSV"),
    effect_col: str = typer.Option("beta",  "--effect-col"),
    pval_col:   str = typer.Option("pval",  "--pval-col"),
    output:     str = typer.Option("volcano.png", "--output", "-o"),
):
    """Volcano plot from GWAS summary statistics."""
    p = plot_volcano(summary_path, output, effect_col=effect_col, pval_col=pval_col)
    typer.echo(f"✅ Volcano plot saved to {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
