"""
feature_selection.py
====================
Feature selection for PredictMix — 15 methods across four families.

Filter (statistical independence):
    none        pass-through, all features
    pearson     Pearson correlation  r(X_j, y)
    chi2        Chi-squared test     χ² = Σ(Oᵢ-Eᵢ)²/Eᵢ
    infogain    Information Gain / mutual information  IG = H(y) - H(y|X_j)

Embedded (regularisation during training):
    lasso       LASSO L1:       Cost = (1/n)Σ(y-ŷ)² + λΣ|w_j|
    ridge       Ridge L2:       Cost = (1/n)Σ(y-ŷ)² + λΣw_j²
    elasticnet  Elastic Net:    Cost = (1/n)Σ(y-ŷ)² + λ[(1-α)|w|₁ + αw²]
    tree        RF importance   Gini-based feature importance

Dimensionality reduction:
    pca         Principal Component Analysis

Wrapper:
    rfe         Recursive Feature Elimination (backward greedy)

Deep learning:
    dnn_l1      L1-DNN:   I_j = Σᵢ|W^(1)_ij|
    gated       Gated:    X̃ = X⊙m ; L = L_task + λ₁‖m‖₁ + λ₂‖W‖²
    autoencoder Sparse AE: h=ReLU(WₑX+bₑ) ; L=‖X-X̂‖²+β‖h‖₁
    stability   Bootstrap: π_j=(1/B)Σ 1(x_j∈X_b) ≥ 0.6
    stacked_dl  Pipeline: X → L1-DNN → X^(1) → Gated → X^(2) → Sparse AE → h
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .config import PredictMixConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Deep learning helpers (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)),
                    np.exp(x)/(1.0+np.exp(x)))

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def _bce_grad(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) / (len(y) + 1e-9)


class _AdamFS:
    """Lightweight Adam for feature selection models."""
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self._m, self._v, self._t = [], [], 0
    def init(self, p):
        self._m = [np.zeros_like(x) for x in p]
        self._v = [np.zeros_like(x) for x in p]
    def step(self, p, g):
        self._t += 1
        for i, (x, gi) in enumerate(zip(p, g)):
            self._m[i] = self.b1*self._m[i] + (1-self.b1)*gi
            self._v[i] = self.b2*self._v[i] + (1-self.b2)*gi**2
            mh = self._m[i]/(1-self.b1**self._t)
            vh = self._v[i]/(1-self.b2**self._t)
            x -= self.lr*(mh/(np.sqrt(vh)+self.eps) + self.wd*x)


# ─────────────────────────────────────────────────────────────────────────────
#  DNN L1 selector
# ─────────────────────────────────────────────────────────────────────────────

def _L1DNNSelector(n_hidden: int = 64, lr: float = 1e-3,
                   lam: float = 0.01, epochs: int = 80,
                   batch_size: int = 64, random_state: int = 42):
    """
    L1-Regularised DNN feature selection.

    Loss:  L_total = BCE + λ·‖W^(1)‖₁
    Importance: I_j = Σᵢ|W^(1)_ij|  — sum of absolute weights in first layer

    Returns importance scores (d,) after training.
    """
    class _Selector:
        def fit(self, X, y):
            rng = np.random.default_rng(random_state)
            n, d = X.shape
            k = np.sqrt(2.0 / d)
            W1 = rng.normal(0, k, (n_hidden, d))
            b1 = np.zeros(n_hidden)
            W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (1, n_hidden))
            b2 = np.zeros(1)
            opt = _AdamFS(lr=lr, wd=0.0)
            opt.init([W1, b1, W2, b2])
            for _ in range(epochs):
                idx = rng.permutation(n)
                for s in range(0, n, batch_size):
                    b = idx[s:s+batch_size]
                    Xb, yb = X[b], y[b]
                    z1 = Xb @ W1.T + b1
                    a1 = _relu(z1)
                    z2 = a1 @ W2.T + b2
                    p  = _sigmoid(z2).ravel()
                    d2 = _bce_grad(p, yb).reshape(-1,1)
                    dW2 = d2.T @ a1 / len(yb)
                    db2 = d2.mean(0)
                    da1 = d2 @ W2
                    dz1 = da1 * _relu_grad(z1)
                    dW1 = dz1.T @ Xb / len(yb) + lam * np.sign(W1)
                    db1 = dz1.mean(0)
                    opt.step([W1, b1, W2, b2], [dW1, db1, dW2, db2])
            return np.abs(W1).sum(0)  # importance per feature
    return _Selector()


# ─────────────────────────────────────────────────────────────────────────────
#  Gated DFS selector
# ─────────────────────────────────────────────────────────────────────────────

def _GatedSelector(n_hidden: int = 64, lam1: float = 1e-3, lam2: float = 1e-4,
                   tau: float = 0.05, lr: float = 1e-3, epochs: int = 80,
                   batch_size: int = 64, random_state: int = 42):
    """
    Gated Feature Selection.

    Learnable mask m ∈ [0,1]^d:   X̃ = X ⊙ m
    Loss:  L_DFS = BCE + λ₁‖m‖₁ + λ₂‖W‖²
    Keep features where m_j > τ.
    """
    class _Selector:
        def fit(self, X, y):
            rng = np.random.default_rng(random_state)
            n, d = X.shape
            k = np.sqrt(2.0 / d)
            m  = np.ones(d) * 0.5
            W1 = rng.normal(0, k, (n_hidden, d))
            b1 = np.zeros(n_hidden)
            W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (1, n_hidden))
            b2 = np.zeros(1)
            opt = _AdamFS(lr=lr, wd=0.0)
            opt.init([m, W1, b1, W2, b2])
            for _ in range(epochs):
                idx = rng.permutation(n)
                for s in range(0, n, batch_size):
                    b = idx[s:s+batch_size]
                    Xb, yb = X[b], y[b]
                    mc   = np.clip(m, 0, 1)
                    Xm   = Xb * mc
                    z1   = Xm @ W1.T + b1
                    a1   = _relu(z1)
                    z2   = a1 @ W2.T + b2
                    p    = _sigmoid(z2).ravel()
                    d2   = _bce_grad(p, yb).reshape(-1,1)
                    dW2  = d2.T @ a1 / len(yb)
                    db2  = d2.mean(0)
                    da1  = d2 @ W2
                    dz1  = da1 * _relu_grad(z1)
                    dW1  = dz1.T @ Xm / len(yb) + lam2 * W1
                    db1  = dz1.mean(0)
                    dXm  = dz1 @ W1
                    dm   = (dXm * Xb).mean(0) + lam1 * np.sign(mc)
                    opt.step([m, W1, b1, W2, b2], [dm, dW1, db1, dW2, db2])
                    m[:] = np.clip(m, 0, 1)
            return m
    return _Selector()


# ─────────────────────────────────────────────────────────────────────────────
#  Sparse Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class _SparseAutoencoder:
    """
    Sparse Autoencoder.

    Encoder:  h = ReLU(Wₑ·X + bₑ)
    Decoder:  X̂ = ReLU(Wₐ·h + bₐ)
    Loss:     L_AE = ‖X - X̂‖² + β·‖h‖₁
    """
    def __init__(self, latent_dim: int = 8, beta: float = 1e-3,
                 lr: float = 1e-3, epochs: int = 80,
                 batch_size: int = 64, random_state: int = 42):
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        ld = self.latent_dim
        ke = np.sqrt(2.0 / d)
        kd = np.sqrt(2.0 / ld)
        We = rng.normal(0, ke, (ld, d))
        be = np.zeros(ld)
        Wd = rng.normal(0, kd, (d, ld))
        bd = np.zeros(d)
        opt = _AdamFS(lr=self.lr, wd=1e-4)
        opt.init([We, be, Wd, bd])
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = idx[s:s+self.batch_size]
                Xb = X[b]
                zh = Xb @ We.T + be
                h  = _relu(zh)
                zr = h @ Wd.T + bd
                r  = _relu(zr)
                # Reconstruction gradient
                dr  = 2.0 * (r - Xb) / len(b)
                dzr = dr * _relu_grad(zr)
                dWd = dzr.T @ h / len(b)
                dbd = dzr.mean(0)
                dh  = dzr @ Wd + self.beta * np.sign(h)
                dzh = dh * _relu_grad(zh)
                dWe = dzh.T @ Xb / len(b)
                dbe = dzh.mean(0)
                opt.step([We, be, Wd, bd], [dWe, dbe, dWd, dbd])
        zh = X @ We.T + be
        return _relu(zh)   # (n, latent_dim)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main select_features function
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: PredictMixConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply the feature selection method specified in cfg.feature_selection.

    Returns
    -------
    (X_selected, column_names)
        X_selected  : pd.DataFrame with selected / transformed features
        column_names: list of column names
    """
    method = cfg.feature_selection
    n_feat = cfg.n_features or X.shape[1]
    rs     = cfg.random_state

    Xa = X.values.astype(float)
    ya = y.values.astype(float)

    # ── none ─────────────────────────────────────────────────────
    if method == "none":
        return X, list(X.columns)

    # ── Pearson Correlation ──────────────────────────────────────
    if method == "pearson":
        # r(X_j, y) = Σ(Xᵢ-X̄)(yᵢ-ȳ) / √[Σ(Xᵢ-X̄)²·Σ(yᵢ-ȳ)²]
        xm = Xa - Xa.mean(0)
        ym = ya - ya.mean()
        num   = (xm * ym[:, None]).sum(0)
        denom = np.sqrt((xm**2).sum(0) * (ym**2).sum() + 1e-9)
        r = np.abs(num / denom)
        selected = np.argsort(r)[::-1][:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Chi-Squared ──────────────────────────────────────────────
    if method == "chi2":
        from sklearn.feature_selection import SelectKBest, chi2
        # Shift to non-negative for chi2
        Xa_pos = Xa - Xa.min(0)
        k = min(n_feat, Xa.shape[1])
        sel = SelectKBest(chi2, k=k).fit(Xa_pos, ya.astype(int))
        mask = sel.get_support()
        cols = [X.columns[i] for i, m in enumerate(mask) if m]
        return X[cols], cols

    # ── Information Gain ─────────────────────────────────────────
    if method == "infogain":
        # IG(X_j, y) = H(y) - H(y|X_j) — mutual information
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        k = min(n_feat, Xa.shape[1])
        sel = SelectKBest(mutual_info_classif, k=k).fit(Xa, ya.astype(int))
        mask = sel.get_support()
        cols = [X.columns[i] for i, m in enumerate(mask) if m]
        return X[cols], cols

    # ── LASSO L1 ─────────────────────────────────────────────────
    if method == "lasso":
        # Cost = (1/n)Σ(y-ŷ)² + λΣ|w_j|
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        lasso = LassoCV(cv=5, random_state=rs, max_iter=5000).fit(Xs, ya)
        imp = np.abs(lasso.coef_)
        selected = np.where(imp > 0)[0]
        if len(selected) == 0:
            selected = np.argsort(imp)[::-1][:n_feat]
        selected = selected[:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Ridge L2 ─────────────────────────────────────────────────
    if method == "ridge":
        # Cost = (1/n)Σ(y-ŷ)² + λΣw_j²
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        # sklearn >= 1.8: use l1_ratio=0 instead of penalty='l2'
        import sklearn as _sk
        from packaging import version as _v
        if _v.parse(_sk.__version__) >= _v.parse("1.8.0"):
            lr_clf = LogisticRegression(l1_ratio=0, solver="lbfgs",
                                        max_iter=2000, random_state=rs)
        else:
            lr_clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                        max_iter=2000, random_state=rs)
        lr_clf.fit(Xs, ya.astype(int))
        imp = np.abs(lr_clf.coef_[0])
        selected = np.argsort(imp)[::-1][:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Elastic Net ──────────────────────────────────────────────
    if method == "elasticnet":
        # Cost = (1/n)Σ(y-ŷ)² + λ[(1-α)|w|₁ + α|w|₂²]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        en = ElasticNetCV(cv=5, random_state=rs, max_iter=5000).fit(Xs, ya)
        imp = np.abs(en.coef_)
        selected = np.where(imp > 0)[0]
        if len(selected) == 0:
            selected = np.argsort(imp)[::-1][:n_feat]
        selected = selected[:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Random Forest importance ─────────────────────────────────
    if method == "tree":
        rf = RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1)
        rf.fit(Xa, ya.astype(int))
        imp = rf.feature_importances_
        selected = np.argsort(imp)[::-1][:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── PCA ──────────────────────────────────────────────────────
    if method == "pca":
        # maximise φ₁₁,…,φₚ₁ {(1/n)Σ(Σⱼ φⱼ₁ xᵢⱼ)²} s.t. Σφ²ⱼ₁=1
        k = min(n_feat, Xa.shape[1], Xa.shape[0] - 1)
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        pca = PCA(n_components=k, random_state=rs).fit(Xs)
        Z = pca.transform(Xs)
        cols = [f"PC{i+1}" for i in range(k)]
        return pd.DataFrame(Z, columns=cols, index=X.index), cols

    # ── RFE (wrapper) ─────────────────────────────────────────────
    if method == "rfe":
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        import sklearn as _sk
        from packaging import version as _v
        k = min(n_feat, Xa.shape[1])
        if _v.parse(_sk.__version__) >= _v.parse("1.8.0"):
            est = LogisticRegression(l1_ratio=0, solver="lbfgs",
                                     max_iter=1000, random_state=rs)
        else:
            est = LogisticRegression(penalty="l2", solver="lbfgs",
                                     max_iter=1000, random_state=rs)
        rfe = RFE(estimator=est, n_features_to_select=k, step=1)
        rfe.fit(Xa, ya.astype(int))
        cols = [X.columns[i] for i, s in enumerate(rfe.support_) if s]
        return X[cols], cols

    # ── L1-DNN ───────────────────────────────────────────────────
    if method == "dnn_l1":
        # I_j = Σᵢ|W^(1)_ij| — importance from sparse input layer
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        imp = _L1DNNSelector(random_state=rs).fit(Xs, ya)
        selected = np.argsort(imp)[::-1][:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Gated DFS ────────────────────────────────────────────────
    if method == "gated":
        # X̃ = X⊙m ; L = L_task + λ₁‖m‖₁ + λ₂‖W‖²
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        mask = _GatedSelector(random_state=rs).fit(Xs, ya)
        selected = np.where(mask > 0.05)[0]
        if len(selected) == 0:
            selected = np.argsort(mask)[::-1][:n_feat]
        selected = selected[:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Sparse Autoencoder ───────────────────────────────────────
    if method == "autoencoder":
        # h=ReLU(WₑX+bₑ) ; L=‖X-X̂‖²+β‖h‖₁
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        lat = min(n_feat, max(4, Xs.shape[1] // 2))
        ae = _SparseAutoencoder(latent_dim=lat, random_state=rs)
        h  = ae.fit_transform(Xs)
        cols = [f"AE{i+1}" for i in range(h.shape[1])]
        return pd.DataFrame(h, columns=cols, index=X.index), cols

    # ── Stability Selection ──────────────────────────────────────
    if method == "stability":
        # π_j = (1/B)Σ 1(x_j ∈ X_b) ≥ 0.6
        B = 50
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        n = len(Xs)
        counts = np.zeros(Xs.shape[1])
        rng = np.random.default_rng(rs)
        for _ in range(B):
            idx = rng.choice(n, size=n // 2, replace=False)
            Xb, yb = Xs[idx], ya[idx]
            lasso = LassoCV(cv=3, max_iter=3000, random_state=rs).fit(Xb, yb)
            counts += (np.abs(lasso.coef_) > 0).astype(float)
        pi = counts / B
        selected = np.where(pi >= 0.6)[0]
        if len(selected) == 0:
            selected = np.argsort(pi)[::-1][:n_feat]
        selected = selected[:n_feat]
        cols = [X.columns[i] for i in selected]
        return X[cols], cols

    # ── Stacked DL pipeline ──────────────────────────────────────
    if method == "stacked_dl":
        # Stage 1: L1-DNN → X^(1)
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        imp1 = _L1DNNSelector(random_state=rs).fit(Xs, ya)
        k1 = min(Xs.shape[1], max(4, int(Xs.shape[1] * 0.75)))
        idx1 = np.argsort(imp1)[::-1][:k1]
        Xs1  = Xs[:, idx1]
        # Stage 2: Gated DFS → X^(2)
        mask2 = _GatedSelector(random_state=rs).fit(Xs1, ya)
        sel2  = np.where(mask2 > 0.05)[0]
        if len(sel2) == 0:
            sel2 = np.argsort(mask2)[::-1][:n_feat]
        Xs2   = Xs1[:, sel2]
        # Stage 3: Sparse Autoencoder → h
        lat = min(n_feat, max(4, Xs2.shape[1] // 2))
        ae  = _SparseAutoencoder(latent_dim=lat, random_state=rs)
        h   = ae.fit_transform(Xs2)
        cols = [f"StkDL{i+1}" for i in range(h.shape[1])]
        return pd.DataFrame(h, columns=cols, index=X.index), cols

    raise ValueError(
        f"Feature selection method '{method}' not recognised.\n"
        "Available: none, pearson, chi2, infogain, lasso, ridge, elasticnet, tree, "
        "pca, rfe, dnn_l1, gated, autoencoder, stability, stacked_dl"
    )
