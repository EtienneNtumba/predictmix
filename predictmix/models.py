"""
models.py
=========
ModelFactory — returns a configured, unfitted sklearn-compatible estimator
for any of the 18 supported model types.

Classical    : logreg, bayesian, svm, rf, mlp, adaboost, bagging, ensemble
Survival     : coxph, deepsurv
Stratif.     : gmm, kmeans_risk
Transfer     : transfer
Deep (NumPy) : rnn, lstm, gru, cnn1d, transformer, sedl
"""
from __future__ import annotations

import sklearn
from packaging import version as pkg_version
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import PredictMixConfig
from .deep_models import (
    CNN1DClassifier,
    CoxPHClassifier,
    DeepSurvClassifier,
    GMMClassifier,
    GRUClassifier,
    KMeansRiskClassifier,
    LSTMClassifier,
    RNNClassifier,
    SEDLClassifier,
    TransferLearningClassifier,
    TransformerClassifier,
)

# Detect sklearn version for API compatibility
_SKLEARN_GE_18 = pkg_version.parse(sklearn.__version__) >= pkg_version.parse("1.8.0")


@dataclass
class ModelFactory:
    """
    Factory that builds a fully configured, unfitted estimator.

    Usage
    -----
    cfg = PredictMixConfig(model="lstm", random_state=42)
    clf = ModelFactory(cfg).build()
    clf.fit(X_train, y_train)
    """
    cfg: PredictMixConfig

    def build(self):
        """Return the configured estimator for cfg.model."""
        m  = self.cfg.model
        rs = self.cfg.random_state

        # ── Logistic Regression ──────────────────────────────────
        if m == "logreg":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    random_state=rs,
                )),
            ])

        # ── Bayesian Classifier (Gaussian Naive Bayes) ───────────
        if m == "bayesian":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ])

        # ── Support Vector Machine ───────────────────────────────
        if m == "svm":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(kernel="rbf", probability=False, random_state=rs),
                    cv=3,
                )),
            ])

        # ── Random Forest ────────────────────────────────────────
        if m == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                random_state=rs,
                n_jobs=-1,
            )

        # ── MLP ──────────────────────────────────────────────────
        if m == "mlp":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=500,
                    early_stopping=True,
                    random_state=rs,
                )),
            ])

        # ── AdaBoost ─────────────────────────────────────────────
        if m == "adaboost":
            # sklearn >= 1.2 deprecated algorithm="SAMME.R"
            # sklearn >= 1.8 removed the algorithm parameter entirely
            if _SKLEARN_GE_18:
                return AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=200,
                    learning_rate=0.5,
                    random_state=rs,
                )
            else:
                return AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1),
                    n_estimators=200,
                    learning_rate=0.5,
                    algorithm="SAMME",
                    random_state=rs,
                )

        # ── Bagging ──────────────────────────────────────────────
        if m == "bagging":
            return BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=5),
                n_estimators=100,
                max_samples=0.8,
                max_features=0.8,
                random_state=rs,
                n_jobs=-1,
            )

        # ── Stacking Ensemble ────────────────────────────────────
        if m == "ensemble":
            base = [
                ("lr", Pipeline([
                    ("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=1000, random_state=rs)),
                ])),
                ("svm", Pipeline([
                    ("s", StandardScaler()),
                    ("c", CalibratedClassifierCV(
                        SVC(kernel="rbf", probability=False, random_state=rs), cv=3
                    )),
                ])),
                ("rf",  RandomForestClassifier(n_estimators=200, random_state=rs)),
                ("ada", AdaBoostClassifier(n_estimators=100, random_state=rs)
                 if _SKLEARN_GE_18 else
                 AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=rs)),
            ]
            meta = LogisticRegression(max_iter=1000, random_state=rs)
            return StackingClassifier(
                estimators=base,
                final_estimator=meta,
                cv=5,
                n_jobs=-1,
            )

        # ── Cox Proportional Hazards ─────────────────────────────
        if m == "coxph":
            return CoxPHClassifier(
                lr=1e-3, epochs=100, batch_size=64,
                weight_decay=1e-4, random_state=rs,
            )

        # ── DeepSurv ─────────────────────────────────────────────
        if m == "deepsurv":
            return DeepSurvClassifier(
                hidden_layers=(128, 64), lr=1e-3, epochs=80,
                batch_size=32, cox_weight=0.5, weight_decay=1e-4,
                random_state=rs,
            )

        # ── Gaussian Mixture Model ───────────────────────────────
        if m == "gmm":
            return GMMClassifier(
                n_components=3, covariance_type="diag", random_state=rs,
            )

        # ── K-Means Risk Classifier ──────────────────────────────
        if m == "kmeans_risk":
            return KMeansRiskClassifier(n_clusters=6, random_state=rs)

        # ── Transfer Learning ────────────────────────────────────
        if m == "transfer":
            return TransferLearningClassifier(
                hidden_sizes=(128, 64),
                pretrain_epochs=60, finetune_epochs=40, finetune_layers=1,
                lr=1e-3, batch_size=32, weight_decay=1e-4, random_state=rs,
            )

        # ── RNN ──────────────────────────────────────────────────
        if m == "rnn":
            return RNNClassifier(
                hidden_size=32, lr=1e-3, epochs=80,
                batch_size=32, weight_decay=1e-4, random_state=rs,
            )

        # ── LSTM ─────────────────────────────────────────────────
        if m == "lstm":
            return LSTMClassifier(
                hidden_size=32, lr=1e-3, epochs=80,
                batch_size=32, weight_decay=1e-4, random_state=rs,
            )

        # ── GRU ──────────────────────────────────────────────────
        if m == "gru":
            return GRUClassifier(
                hidden_size=32, lr=1e-3, epochs=80,
                batch_size=32, weight_decay=1e-4, random_state=rs,
            )

        # ── 1-D CNN ──────────────────────────────────────────────
        if m == "cnn1d":
            return CNN1DClassifier(
                num_filters=32, filter_sizes=(2, 3, 4), fc_hidden=64,
                lr=1e-3, epochs=80, batch_size=32, weight_decay=1e-4,
                random_state=rs,
            )

        # ── Transformer ──────────────────────────────────────────
        if m == "transformer":
            return TransformerClassifier(
                d_model=32, n_heads=4, d_ff=64,
                lr=1e-3, epochs=80, batch_size=32, weight_decay=1e-4,
                random_state=rs,
            )

        # ── SEDL – Stacked Ensemble Deep Learning ─────────────────
        if m == "sedl":
            return SEDLClassifier(cv_folds=5, random_state=rs)

        raise ValueError(
            f"Unknown model: '{m}'.\n"
            "Available: logreg, bayesian, svm, rf, mlp, adaboost, bagging, ensemble, "
            "coxph, deepsurv, gmm, kmeans_risk, transfer, "
            "rnn, lstm, gru, cnn1d, transformer, sedl"
        )
