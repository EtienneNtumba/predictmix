from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier

from .config import PredictMixConfig

def select_features(
    X: pd.DataFrame, y: pd.Series, cfg: PredictMixConfig
) -> Tuple[pd.DataFrame, list[str]]:
    method = cfg.feature_selection
    if method == "none":
        return X, list(X.columns)

    if method == "chi2":
        scores, _ = chi2(X, y)
        idx = np.argsort(scores)[::-1][: cfg.n_features]
        cols = X.columns[idx]
        return X[cols], list(cols)

    if method in {"lasso", "elasticnet"}:
        if method == "lasso":
            model = LassoCV(cv=5, random_state=cfg.random_state)
        else:
            model = ElasticNetCV(cv=5, random_state=cfg.random_state, l1_ratio=0.5)

        model.fit(X, y)
        coef = np.abs(model.coef_)
        idx = np.argsort(coef)[::-1][: cfg.n_features]
        cols = X.columns[idx]
        return X[cols], list(cols)

    if method == "tree":
        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1][: cfg.n_features]
        cols = X.columns[idx]
        return X[cols], list(cols)

    if method == "pca":
        n_comp = min(cfg.n_features or X.shape[1], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=cfg.random_state)
        X_pca = pca.fit_transform(X)
        cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=cols, index=X.index), cols

    raise ValueError(f"Feature selection method not supported: {method}")

