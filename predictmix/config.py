"""
config.py
=========
PredictMixConfig — central configuration dataclass for PredictMix.

18 prediction models:
  Classical   : logreg, bayesian, svm, rf, mlp, adaboost, bagging, ensemble
  Survival    : coxph, deepsurv
  Stratif.    : gmm, kmeans_risk
  Transfer    : transfer
  Deep (NumPy): rnn, lstm, gru, cnn1d, transformer, sedl

15 feature selection methods:
  Filter   : none, pearson, chi2, infogain
  Embedded : lasso, ridge, elasticnet, tree
  Reduction: pca
  Wrapper  : rfe
  Deep DL  : dnn_l1, gated, autoencoder, stability, stacked_dl
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from typing_extensions import Literal

ModelName = Literal[
    # Classical / statistical
    "logreg", "bayesian", "svm", "rf", "mlp",
    "adaboost", "bagging", "ensemble",
    # Survival analysis
    "coxph", "deepsurv",
    # Risk stratification
    "gmm", "kmeans_risk",
    # Transfer learning
    "transfer",
    # Deep learning (pure NumPy)
    "rnn", "lstm", "gru", "cnn1d", "transformer", "sedl",
]

FSMethod = Literal[
    # Filter methods
    "none", "pearson", "chi2", "infogain",
    # Embedded methods
    "lasso", "ridge", "elasticnet", "tree",
    # Dimensionality reduction
    "pca",
    # Wrapper methods
    "rfe",
    # Deep learning feature selection
    "dnn_l1", "gated", "autoencoder", "stability", "stacked_dl",
]


@dataclass
class PredictMixConfig:
    # ── Target & identifiers ──────────────────────────────────────
    target_column: str = "y"
    id_column: Optional[str] = None

    # ── PRS integration ───────────────────────────────────────────
    prs_column: Optional[str] = "prs"
    genotype_prefix: Optional[str] = None
    beta_file: Optional[str] = None

    # ── Feature selection ─────────────────────────────────────────
    feature_selection: FSMethod = "lasso"
    n_features: Optional[int] = 100

    # ── Model ─────────────────────────────────────────────────────
    model: ModelName = "ensemble"

    # ── Training ──────────────────────────────────────────────────
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.2

    # ── Output ────────────────────────────────────────────────────
    output_dir: str = "predictmix_output"
    drop_columns: List[str] = field(default_factory=list)
