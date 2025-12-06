from __future__ import annotations


from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import PredictMixConfig
from .pipeline import PredictMixPipeline
from .data import load_dataset
from .plots import (
    plot_curves,
    plot_all_from_results,
    plot_volcano,
)
from . import __version__, __credits__



app = typer.Typer(
    help=(
        "PredictMix: integrated disease risk prediction pipeline combining PRS, "
        "clinical, and other risk factors.\n\n"
        "Developed by Etienne Ntumba Kabongo (McGill University) "
        "and Prof. Dr Emile Chimusa (University of Northumbria)."
    )
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show PredictMix version and exit.",
    ),
):
    """
    PredictMix – a modular pipeline for disease risk prediction.

    Authors:
      - Etienne Ntumba Kabongo (McGill University)
      - Prof. Dr Emile Chimusa (University of Northumbria)
    """
    if version:
        typer.echo(f"predictmix version {__version__}")
        typer.echo("Authors:")
        for c in __credits__:
            typer.echo(f"  - {c}")
        raise typer.Exit()


@app.command()
def train(
    data: str = typer.Argument(
        ...,
        help="Path to the input dataset (CSV/Parquet) containing the target column.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "YAML configuration file. If provided, model and feature-selection options "
            "from the CLI are ignored."
        ),
    ),
    model: str = typer.Option(
        "ensemble",
        "--model",
        "-m",
        help="Prediction model: logreg, svm, rf, mlp, ensemble.",
    ),
    feature_selection: str = typer.Option(
        "lasso",
        "--feature-selection",
        "-f",
        help="Feature selection method: none, lasso, elasticnet, tree, chi2, pca.",
    ),
    n_features: int = typer.Option(
        100,
        "--n-features",
        "-k",
        help="Number of features to keep after feature selection.",
    ),
    target_column: str = typer.Option(
        "y",
        "--target-column",
        "-y",
        help="Name of the binary target column (0/1).",
    ),
    output_dir: str = typer.Option(
        "predictmix_output",
        "--output-dir",
        "-o",
        help="Output directory for the model, config and metrics.",
    ),
    export_predictions: Optional[str] = typer.Option(
        None,
        "--export-predictions",
        help=(
            "Optional CSV path to save per-individual predictions with columns "
            "'y_true', 'risk_proba', and 'split' (train_cv/test). "
            "If not provided, defaults to <output_dir>/predictions.csv."
        ),
    ),
    plots: bool = typer.Option(
        False,
        "--plots/--no-plots",
        help=(
            "If enabled, automatically generate ROC and Precision–Recall curves "
            "from the exported predictions."
        ),
    ),
):
    """
    Train a PredictMix model on a dataset and compute performance metrics
    (cross-validation + held-out test set). Optionally export predictions
    and generate ROC/PR plots.
    """
    # Load config from YAML if provided
    if config:
        with open(config) as f:
            cfg_dict = yaml.safe_load(f)
        cfg = PredictMixConfig(**cfg_dict)
    else:
        cfg = PredictMixConfig(
            target_column=target_column,
            feature_selection=feature_selection,
            n_features=n_features,
            model=model,
            output_dir=output_dir,
        )

    pipe = PredictMixPipeline(cfg)

    # Default export path if not specified
    if export_predictions is None:
        export_predictions_path = str(Path(cfg.output_dir) / "predictions.csv")
    else:
        export_predictions_path = export_predictions

    metrics = pipe.fit(data, export_predictions=export_predictions_path)
    pipe.save()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON
    import json

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Pretty summary
    typer.echo(f"\n=== PredictMix training finished ===")
    typer.echo(f"Model:             {cfg.model}")
    typer.echo(
        f"Feature selection: {cfg.feature_selection} (n_features={cfg.n_features})"
    )
    typer.echo(f"Output dir:        {cfg.output_dir}")

    cv = metrics["cv"]
    te = metrics["test"]

    typer.echo("\nCross-validation metrics:")
    typer.echo(
        f"  AUC={cv['auc']:.3f} | Acc={cv['accuracy']:.3f} | "
        f"F1_macro={cv['f1_macro']:.3f}"
    )

    typer.echo("Test set metrics:")
    typer.echo(
        f"  AUC={te['auc']:.3f} | Acc={te['accuracy']:.3f} | "
        f"F1_macro={te['f1_macro']:.3f}"
    )

    # Optional ROC/PR plots
    if plots:
        plots_dir = str(Path(cfg.output_dir) / "plots")
        paths = plot_curves(export_predictions_path, plots_dir)
        typer.echo("\nGenerated plots:")
        typer.echo(f"  ROC curve: {paths['roc']}")
        typer.echo(f"  PR curve:  {paths['pr']}")

    typer.echo(f"\nModel and configuration saved to: {out_dir}")


@app.command()
def predict(
    model: str = typer.Argument(
        ...,
        help="Path to a trained model file (predictmix_model.joblib).",
    ),
    data: str = typer.Argument(
        ...,
        help="CSV/Parquet file with new individuals (target column not required).",
    ),
    output: str = typer.Option(
        "predictmix_predictions.csv",
        "--output",
        "-o",
        help="Output CSV file for predictions (adds a 'risk_proba' column).",
    ),
):
    """
    Apply a trained PredictMix model to new individuals and export risk
    probabilities for each row.
    """
    pipe = PredictMixPipeline.load(model)
    df = load_dataset(data)
    proba = pipe.predict_proba(df)

    out_df = df.copy()
    out_df["risk_proba"] = proba
    out_df.to_csv(output, index=False)
    typer.echo(f"Predictions saved to: {output}")



@app.command()
def plot(
    results: str = typer.Argument(
        ...,
        help=(
            "CSV file with columns 'y_true' (0/1) and 'risk_proba' "
            "(predicted probability)."
        ),
    ),
    kind: str = typer.Option(
        "all",
        "--kind",
        "-k",
        help=(
            "Type of plot to generate: "
            "rocpr, hist, scatter, heatmap, calib, all."
        ),
    ),
    output_dir: str = typer.Option(
        "predictmix_plots",
        "--output-dir",
        "-o",
        help="Output directory for the generated plots.",
    ),
):
    """
    Generate one or several plots from a results CSV file containing
    true labels and predicted probabilities.
    """
    paths = plot_all_from_results(results, output_dir, kind=kind)
    typer.echo("Generated plots:")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")

@app.command(name="plot-volcano")
def plot_volcano_cmd(
    summary: str = typer.Argument(
        ...,
        help=(
            "GWAS-like summary statistics CSV with effect and p-value columns "
            "(e.g., beta, pval)."
        ),
    ),
    effect_col: str = typer.Option(
        "beta",
        "--effect-col",
        help="Name of the effect size column (e.g., beta, logOR).",
    ),
    pval_col: str = typer.Option(
        "pval",
        "--pval-col",
        help="Name of the p-value column.",
    ),
    output: str = typer.Option(
        "predictmix_volcano.png",
        "--output",
        "-o",
        help="Path to the output PNG file for the volcano plot.",
    ),
):
    """
    Generate a volcano plot from GWAS-like summary statistics.
    """
    path = plot_volcano(
        summary_path=summary,
        output_path=output,
        effect_col=effect_col,
        pval_col=pval_col,
    )
    typer.echo(f"Volcano plot saved to: {path}")

