from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

from .config import PredictMixConfig
from .data import load_dataset, split_xy
from .prs import compute_prs_from_genotypes
from .feature_selection import select_features
from .models import ModelFactory


class PredictMixPipeline:
    def __init__(self, cfg: PredictMixConfig):
        self.cfg = cfg
        self.model = None
        self.selected_features = []

    def fit(
        self,
        data_path: str,
        export_predictions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle, retourne les métriques CV + test.
        Si export_predictions est fourni, sauvegarde un CSV avec:
          - y_true
          - risk_proba
          - split (train/test)
        pour pouvoir faire des plots et des analyses downstream.
        """
        df = load_dataset(data_path)
        df = compute_prs_from_genotypes(df, self.cfg)

        X, y = split_xy(df, self.cfg)
        X_fs, cols = select_features(X, y, self.cfg)
        self.selected_features = cols

        X_train, X_test, y_train, y_test = train_test_split(
            X_fs,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

        model = ModelFactory(self.cfg).build()

        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )

        y_proba_cv = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        y_pred_cv = (y_proba_cv >= 0.5).astype(int)
        metrics_cv = self._compute_metrics(y_train, y_pred_cv, y_proba_cv)

        # fit final
        model.fit(X_train, y_train)
        self.model = model

        # test set
        y_proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= 0.5).astype(int)
        metrics_test = self._compute_metrics(y_test, y_pred_test, y_proba_test)

        # export prédictions si demandé
        if export_predictions is not None:
            export_path = Path(export_predictions)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            df_pred_train = pd.DataFrame(
                {
                    "y_true": y_train.values,
                    "risk_proba": y_proba_cv,
                    "split": "train_cv",
                }
            )
            df_pred_test = pd.DataFrame(
                {
                    "y_true": y_test.values,
                    "risk_proba": y_proba_test,
                    "split": "test",
                }
            )

            df_all = pd.concat([df_pred_train, df_pred_test], axis=0, ignore_index=True)
            df_all.to_csv(export_path, index=False)

        return {
            "cv": metrics_cv,
            "test": metrics_test,
        }

    def _compute_metrics(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy": float(acc),
            "auc": float(auc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        }

    def save(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "cfg": self.cfg,
                "model": self.model,
                "selected_features": self.selected_features,
            },
            out / "predictmix_model.joblib",
        )

        with open(out / "config.json", "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PredictMixPipeline":
        obj = joblib.load(path)
        pipe = cls(obj["cfg"])
        pipe.model = obj["model"]
        pipe.selected_features = obj["selected_features"]
        return pipe

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        df = compute_prs_from_genotypes(df, self.cfg)
        X, _ = split_xy(df, self.cfg)
        X = X[self.selected_features]
        return self.model.predict_proba(X)[:, 1]

