"""
pipeline.py
===========
PredictMixPipeline — training, cross-validation, evaluation, persistence,
and prediction for PredictMix v0.2.0.

Supports all 18 model types:
  - Classical sklearn models: standard cross_val_predict with n_jobs=-1
  - Deep learning models (rnn,lstm,gru,cnn1d,transformer,deepsurv,sedl,
    coxph,transfer): manual StratifiedKFold with copy.deepcopy per fold
  - Stratification models (gmm,kmeans_risk): manual KFold
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from .config import PredictMixConfig
from .data import load_dataset, split_xy
from .feature_selection import select_features
from .models import ModelFactory
from .prs import compute_prs_from_genotypes

# Models that require manual CV (no cross_val_predict support)
_MANUAL_CV_MODELS = {
    "rnn", "lstm", "gru", "cnn1d", "transformer",
    "deepsurv", "sedl", "coxph", "transfer",
    "gmm", "kmeans_risk",
}


class PredictMixPipeline:
    """
    End-to-end PredictMix pipeline:
        load data → PRS → feature selection → train → CV evaluate → save

    Usage
    -----
    cfg = PredictMixConfig(model="lstm", feature_selection="lasso")
    pipe = PredictMixPipeline(cfg)
    metrics = pipe.fit("data.csv", export_predictions="results/preds.csv")
    pipe.save()

    # Later — inference
    pipe2 = PredictMixPipeline.load("output/predictmix_model.joblib")
    probas = pipe2.predict_proba(new_df)
    """

    def __init__(self, cfg: PredictMixConfig):
        self.cfg = cfg
        self.model = None
        self.selected_features: list = []

    # ── Training entry-point ─────────────────────────────────────

    def fit(
        self,
        data_path: str,
        export_predictions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model, run cross-validation, evaluate on held-out test set.

        Parameters
        ----------
        data_path          : path to CSV / Parquet file
        export_predictions : optional path for per-sample predictions CSV
                             (columns: y_true, risk_proba, split)

        Returns
        -------
        {"cv": {...}, "test": {...}}  — metric dicts
        """
        df = load_dataset(data_path)
        df = compute_prs_from_genotypes(df, self.cfg)

        X, y = split_xy(df, self.cfg)
        X_fs, cols = select_features(X, y, self.cfg)
        self.selected_features = cols

        X_train, X_test, y_train, y_test = train_test_split(
            X_fs, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

        model = ModelFactory(self.cfg).build()

        # Cross-validation
        if self.cfg.model in _MANUAL_CV_MODELS:
            metrics_cv, y_proba_cv = self._cv_manual(model, X_train, y_train)
        else:
            metrics_cv, y_proba_cv = self._cv_sklearn(model, X_train, y_train)

        # Final fit on full training set
        model.fit(X_train, y_train)
        self.model = model

        # Test-set evaluation
        y_proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_test  = (y_proba_test >= 0.5).astype(int)
        metrics_test = self._compute_metrics(y_test, y_pred_test, y_proba_test)

        if export_predictions:
            self._export_preds(export_predictions,
                               y_train, y_proba_cv,
                               y_test, y_proba_test)

        return {"cv": metrics_cv, "test": metrics_test}

    # ── sklearn cross_val_predict ────────────────────────────────

    def _cv_sklearn(self, model, X_train, y_train):
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds, shuffle=True,
            random_state=self.cfg.random_state,
        )
        y_proba_cv = cross_val_predict(
            model, X_train, y_train,
            cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        y_pred_cv = (y_proba_cv >= 0.5).astype(int)
        return self._compute_metrics(y_train, y_pred_cv, y_proba_cv), y_proba_cv

    # ── Manual CV (deep / stratification models) ──────────────────

    def _cv_manual(self, model_template, X_train, y_train):
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds, shuffle=True,
            random_state=self.cfg.random_state,
        )
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
        y_proba_cv = np.zeros(len(y_arr))
        for tr_idx, val_idx in cv.split(X_arr, y_arr):
            m = copy.deepcopy(model_template)
            m.fit(X_arr[tr_idx], y_arr[tr_idx])
            y_proba_cv[val_idx] = m.predict_proba(X_arr[val_idx])[:, 1]
        y_pred_cv = (y_proba_cv >= 0.5).astype(int)
        return self._compute_metrics(y_arr, y_pred_cv, y_proba_cv), y_proba_cv

    # ── Metrics ──────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = float("nan")
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy":        float(acc),
            "auc":             float(auc),
            "precision_macro": float(prec),
            "recall_macro":    float(rec),
            "f1_macro":        float(f1),
        }

    # ── Export predictions CSV ────────────────────────────────────

    @staticmethod
    def _export_preds(path, y_train, proba_train, y_test, proba_test):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat([
            pd.DataFrame({"y_true": np.asarray(y_train),
                          "risk_proba": proba_train, "split": "train_cv"}),
            pd.DataFrame({"y_true": np.asarray(y_test),
                          "risk_proba": proba_test, "split": "test"}),
        ], ignore_index=True)
        df.to_csv(out, index=False)

    # ── Persistence ───────────────────────────────────────────────

    def save(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "cfg":               self.cfg,
            "model":             self.model,
            "selected_features": self.selected_features,
        }, out / "predictmix_model.joblib")
        with open(out / "config.json", "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PredictMixPipeline":
        obj = joblib.load(path)
        pipe = cls(obj["cfg"])
        pipe.model = obj["model"]
        pipe.selected_features = obj["selected_features"]
        return pipe

    # ── Inference ─────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        df = compute_prs_from_genotypes(df, self.cfg)
        if self.cfg.target_column in df.columns:
            X, _ = split_xy(df, self.cfg)
        else:
            drop = [c for c in self.cfg.drop_columns if c in df.columns]
            X = df.drop(columns=drop)
        # Keep only features seen during training
        named = [f for f in self.selected_features if f in X.columns]
        if named:
            X = X[named]
        return self.model.predict_proba(X)[:, 1]
