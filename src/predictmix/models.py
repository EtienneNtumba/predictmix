from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PredictMixConfig

@dataclass
class ModelFactory:
    cfg: PredictMixConfig

    def build(self):
        m = self.cfg.model

        if m == "logreg":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    max_iter=500,
                    n_jobs=-1,
                )),
            ])

        if m == "svm":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True)),
            ])

        if m == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )

        if m == "mlp":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=200,
                    random_state=self.cfg.random_state,
                )),
            ])

        if m == "ensemble":
            base_estimators = [
                ("lr", Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, n_jobs=-1)),
                ])),
                ("svm", Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True)),
                ])),
                ("rf", RandomForestClassifier(
                    n_estimators=400,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )),
            ]
            final_estimator = LogisticRegression(max_iter=500)
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=final_estimator,
                stack_method="predict_proba",
                n_jobs=-1,
            )

        raise ValueError(f"Unknown model: {m}")

