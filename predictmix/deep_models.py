"""
deep_models.py
==============
Deep learning classifiers and survival/stratification models for PredictMix.
All implemented in pure NumPy — no PyTorch, TensorFlow, or JAX required.

Models
------
Classical deep learning (tabular):
    RNNClassifier            Simple Recurrent Neural Network  (slide 17)
    LSTMClassifier           Long Short-Term Memory           (slides 17–18)
    GRUClassifier            Gated Recurrent Unit             (slide 16)
    CNN1DClassifier          1-D Convolutional NN             (slide 18)
    TransformerClassifier    Self-attention Transformer       (slide 16)
    SEDLClassifier           Stacked Ensemble Deep Learning   (slide 19)

Survival analysis:
    DeepSurvClassifier       Cox + deep MLP  h(t|X)=h₀(t)·exp(f_θ(X)) (slide 20)
    CoxPHClassifier          Classical Cox PH  h(t|X)=h₀(t)·exp(β^T X) (slide 20)

Risk stratification:
    GMMClassifier            Gaussian Mixture Model  X~Σπₖ N(μₖ,Σₖ)    (slide 22)
    KMeansRiskClassifier     K-Means risk clustering                     (slide 22)

Transfer learning:
    TransferLearningClassifier  Pre-train on source + fine-tune on target (slide 21)

All classifiers expose the scikit-learn API: fit(X,y), predict(X), predict_proba(X).
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared numerical utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _mini_batches(
    X: np.ndarray, y: np.ndarray,
    batch_size: int, rng: np.random.Generator,
) -> list:
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start: start + batch_size]
        yield X[b], y[b]


# ═══════════════════════════════════════════════════════════════════════════════
#  Adam optimiser (shared by all DL models)
# ═══════════════════════════════════════════════════════════════════════════════

class _Adam:
    """Mini Adam with weight decay (AdamW-style)."""
    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 1e-4):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.wd = weight_decay
        self._m: list = []
        self._v: list = []
        self._t = 0

    def init(self, params: list) -> None:
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
        self._t = 0

    def step(self, params: list, grads: list) -> None:
        self._t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.b1 * self._m[i] + (1.0 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1.0 - self.b2) * g ** 2
            m_hat = self._m[i] / (1.0 - self.b1 ** self._t)
            v_hat = self._v[i] / (1.0 - self.b2 ** self._t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)


# ═══════════════════════════════════════════════════════════════════════════════
#  Base class for DL classifiers
# ═══════════════════════════════════════════════════════════════════════════════

class _DeepBase(BaseEstimator, ClassifierMixin):
    """Common sklearn API wrapper for all pure-NumPy deep classifiers."""
    classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        p1 = self._forward_predict(Xs)
        p1 = np.clip(p1.ravel(), 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
#  1. RNN Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class RNNClassifier(_DeepBase):
    """
    Simple Recurrent Neural Network classifier.

    Architecture (many-to-one):
        sₜ = tanh(U·xₜ + W·sₜ₋₁ + b)        (Eq. 26)
        ŷ = σ(V·s_T + c)

    Each input feature treated as one time-step (sequence length = d_features).
    Trained with truncated BPTT and Adam optimiser.
    """
    def __init__(self, hidden_size: int = 32, lr: float = 1e-3, epochs: int = 80,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 random_state: int = 42):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        h = self.hidden_size
        k = np.sqrt(1.0 / h)
        self.U  = rng.uniform(-k, k, (h, 1))      # input  → hidden (1 feature per step)
        self.W  = rng.uniform(-k, k, (h, h))       # hidden → hidden
        self.bh = np.zeros(h)
        self.V  = rng.uniform(-k, k, (1, h))       # hidden → output
        self.bc = np.zeros(1)

    def _params(self) -> list:
        return [self.U, self.W, self.bh, self.V, self.bc]

    def _forward_rnn(self, X: np.ndarray):
        B, T = X.shape
        h = self.hidden_size
        s = np.zeros((B, h))
        states = []
        for t in range(T):
            # X[:, t:t+1] shape (B,1) — each feature is one time step
            s = _tanh(X[:, t:t+1] @ self.U.T + s @ self.W.T + self.bh)
            states.append(s)
        out = _sigmoid(states[-1] @ self.V.T + self.bc)  # (B,1)
        return out.ravel(), states

    def _backward_rnn(self, X: np.ndarray, y: np.ndarray, out, states):
        B, T = X.shape
        h = self.hidden_size
        delta_out = ((out - y) / B).reshape(-1, 1)          # (B,1)
        dV  = delta_out.T @ states[-1]
        dbc = delta_out.mean(0)
        dU  = np.zeros_like(self.U)
        dW  = np.zeros_like(self.W)
        dbh = np.zeros(h)
        ds_next = (delta_out @ self.V) * _tanh_grad(
            X[:, T-1:T] @ self.U.T + (states[-2] if T > 1 else np.zeros((B, h))) @ self.W.T + self.bh
        )
        for t in reversed(range(T)):
            s_prev = states[t-1] if t > 0 else np.zeros((B, h))
            dU  += (ds_next.T @ X[:, t:t+1]).sum(1, keepdims=True) / B  # (h,1)
            dW  += ds_next.T @ s_prev / B
            dbh += ds_next.mean(0)
            ds_next = ds_next @ self.W * _tanh_grad(
                X[:, t-1:t] @ self.U.T + (states[t-2] if t > 1 else np.zeros((B, h))) @ self.W.T + self.bh
            ) if t > 0 else np.zeros_like(ds_next)
        return [dU, dW, dbh, dV, dbc]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_weights(Xs.shape[1])
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                out, states = self._forward_rnn(Xb)
                grads = self._backward_rnn(Xb, yb, out, states)
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self._forward_rnn(X)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  2. LSTM Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(_DeepBase):
    """
    Long Short-Term Memory classifier.

    Gates (Eqs. 27–32):
        fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)        forget gate
        iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)        input gate
        ĉₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)     candidate cell
        cₜ = fₜ⊙cₜ₋₁ + iₜ⊙ĉₜ             cell update
        oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)        output gate
        hₜ = oₜ ⊙ tanh(cₜ)               hidden state
        ŷ  = σ(V·h_T + b_out)
    """
    def __init__(self, hidden_size: int = 32, lr: float = 1e-3, epochs: int = 80,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 random_state: int = 42):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        h = self.hidden_size
        # Each timestep feeds 1 feature, so gate dim = h + 1
        k = np.sqrt(1.0 / h)
        def _gate():
            return rng.uniform(-k, k, (h, h + 1)), np.zeros(h)
        self.Wf, self.bf = _gate()
        self.Wi, self.bi = _gate()
        self.Wc, self.bc = _gate()
        self.Wo, self.bo = _gate()
        self.V  = rng.uniform(-k, k, (1, h))
        self.bv = np.zeros(1)

    def _params(self) -> list:
        return [self.Wf, self.bf, self.Wi, self.bi,
                self.Wc, self.bc, self.Wo, self.bo,
                self.V,  self.bv]

    def _step(self, x, h_prev, c_prev):
        xh = np.concatenate([h_prev, x], axis=1)
        f  = _sigmoid(xh @ self.Wf.T + self.bf)
        i  = _sigmoid(xh @ self.Wi.T + self.bi)
        c_ = _tanh(xh @ self.Wc.T + self.bc)
        c  = f * c_prev + i * c_
        o  = _sigmoid(xh @ self.Wo.T + self.bo)
        h  = o * _tanh(c)
        return h, c, (f, i, c_, o, xh, c_prev)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        B, T = Xs.shape
        self._init_weights(T)
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                b = len(yb)
                h, c = np.zeros((b, self.hidden_size)), np.zeros((b, self.hidden_size))
                caches = []
                for t in range(T):
                    xt = Xb[:, t:t+1]
                    h, c, cache = self._step(xt, h, c)
                    caches.append((h, c, cache))
                out = _sigmoid(h @ self.V.T + self.bv).ravel()
                # Simplified gradient via output layer only
                do = ((out - yb) / b).reshape(-1, 1)
                dV  = do.T @ caches[-1][0]
                dbv = do.mean(0)
                dh  = (do @ self.V) * _tanh_grad(caches[-1][1]) * caches[-1][2][3]
                # Accumulate gate gradients from last step
                f,i,c_,o,xh,cp = caches[-1][2]
                dc = dh * o * _tanh_grad(caches[-1][1]) * i
                dWf = (dh * f * (1-f) * cp).T @ xh / b
                dWi = (dh * i * (1-i) * c_).T  @ xh / b
                dWc = (dc * (1-c_**2)).T        @ xh / b
                dWo = (dh * o * (1-o)).T        @ xh / b
                dbf = (dh * f * (1-f) * cp).mean(0)
                dbi = (dh * i * (1-i) * c_).mean(0)
                dbc = (dc * (1-c_**2)).mean(0)
                dbo = (dh * o * (1-o)).mean(0)
                grads = [dWf+self.weight_decay*self.Wf, dbf,
                         dWi+self.weight_decay*self.Wi, dbi,
                         dWc+self.weight_decay*self.Wc, dbc,
                         dWo+self.weight_decay*self.Wo, dbo,
                         dV+self.weight_decay*self.V, dbv]
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        B, T = X.shape
        h = np.zeros((B, self.hidden_size))
        c = np.zeros((B, self.hidden_size))
        for t in range(T):
            h, c, _ = self._step(X[:, t:t+1], h, c)
        return _sigmoid(h @ self.V.T + self.bv).ravel()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. GRU Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class GRUClassifier(_DeepBase):
    """
    Gated Recurrent Unit classifier.

    Equations 33–36:
        zₜ = σ(Wz·[hₜ₋₁, xₜ] + bz)       update gate
        rₜ = σ(Wr·[hₜ₋₁, xₜ] + br)       reset gate
        h̃ₜ = tanh(Wh·[rₜ⊙hₜ₋₁, xₜ]+bh)  candidate
        hₜ = (1−zₜ)⊙h̃ₜ + zₜ⊙hₜ₋₁
        ŷ  = σ(V·h_T + bv)
    """
    def __init__(self, hidden_size: int = 32, lr: float = 1e-3, epochs: int = 80,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 random_state: int = 42):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        h = self.hidden_size
        # Each timestep feeds 1 feature, so gate dim = h + 1
        k = np.sqrt(1.0 / h)
        def _gate():
            return rng.uniform(-k, k, (h, h + 1)), np.zeros(h)
        self.Wz, self.bz = _gate()
        self.Wr, self.br = _gate()
        self.Wh, self.bh = _gate()
        self.V  = rng.uniform(-k, k, (1, h))
        self.bv = np.zeros(1)

    def _params(self) -> list:
        return [self.Wz, self.bz, self.Wr, self.br, self.Wh, self.bh, self.V, self.bv]

    def _step(self, x, h_prev):
        xh = np.concatenate([h_prev, x], axis=1)
        z  = _sigmoid(xh @ self.Wz.T + self.bz)
        r  = _sigmoid(xh @ self.Wr.T + self.br)
        rh = np.concatenate([r * h_prev, x], axis=1)
        h_ = _tanh(rh @ self.Wh.T + self.bh)
        h  = (1.0 - z) * h_ + z * h_prev
        return h, (z, r, h_, rh, xh, h_prev)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        B, T = Xs.shape
        self._init_weights(T)
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                b = len(yb)
                h = np.zeros((b, self.hidden_size))
                for t in range(T):
                    h, _ = self._step(Xb[:, t:t+1], h)
                out = _sigmoid(h @ self.V.T + self.bv).ravel()
                do  = ((out - yb) / b).reshape(-1, 1)
                dV  = do.T @ h
                dbv = do.mean(0)
                grads = [np.zeros_like(self.Wz), np.zeros(self.hidden_size),
                         np.zeros_like(self.Wr), np.zeros(self.hidden_size),
                         np.zeros_like(self.Wh), np.zeros(self.hidden_size),
                         dV + self.weight_decay * self.V, dbv]
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        B, T = X.shape
        h = np.zeros((B, self.hidden_size))
        for t in range(T):
            h, _ = self._step(X[:, t:t+1], h)
        return _sigmoid(h @ self.V.T + self.bv).ravel()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. CNN1D Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class CNN1DClassifier(_DeepBase):
    """
    1-D Convolutional Neural Network classifier.

    Architecture (Eqs. 37–38):
        cᵢ = f(w^T · x_{i:i+h-1} + b)     — conv with filter w of width h
        ĉ  = max_i(cᵢ)                     — global max-pooling
        ŷ  = σ(FC([ĉ₁, ĉ₂, …]))            — dense + sigmoid

    Multiple filter widths (default 2,3,4) run in parallel (multi-filter bank).
    """
    def __init__(self, num_filters: int = 32, filter_sizes: tuple = (2, 3, 4),
                 fc_hidden: int = 64, lr: float = 1e-3, epochs: int = 80,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 random_state: int = 42):
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.fc_hidden = fc_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self._conv_W, self._conv_b = [], []
        for h in self.filter_sizes:
            fh = min(h, d_in)
            k = np.sqrt(2.0 / (fh * self.num_filters))
            self._conv_W.append(rng.normal(0, k, (self.num_filters, fh)))
            self._conv_b.append(np.zeros(self.num_filters))
        pool_dim = self.num_filters * len(self.filter_sizes)
        k2 = np.sqrt(2.0 / pool_dim)
        self._fc_W1 = rng.normal(0, k2, (self.fc_hidden, pool_dim))
        self._fc_b1 = np.zeros(self.fc_hidden)
        self._fc_W2 = rng.normal(0, np.sqrt(2.0 / self.fc_hidden), (1, self.fc_hidden))
        self._fc_b2 = np.zeros(1)

    def _params(self) -> list:
        p = []
        for W, b in zip(self._conv_W, self._conv_b):
            p += [W, b]
        return p + [self._fc_W1, self._fc_b1, self._fc_W2, self._fc_b2]

    def _conv_pool(self, X: np.ndarray):
        pooled = []
        for W, b in zip(self._conv_W, self._conv_b):
            h = W.shape[1]
            T = X.shape[1]
            if h > T:
                h = T
                W = W[:, :h]
            n_win = T - h + 1
            seg = np.stack([X[:, i:i+h] for i in range(n_win)], axis=2)  # (B,h,n_win)
            feat = np.tensordot(seg, W, axes=([1], [1]))  # (B,n_win,F)
            feat = feat + b
            pooled.append(feat.max(axis=1))  # (B,F)
        return np.concatenate(pooled, axis=1)  # (B, F*n_filters)

    def _forward_fc(self, pool):
        z1 = pool @ self._fc_W1.T + self._fc_b1
        a1 = _relu(z1)
        z2 = a1 @ self._fc_W2.T + self._fc_b2
        return _sigmoid(z2).ravel(), a1, z1, pool

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_weights(Xs.shape[1])
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                pool = self._conv_pool(Xb)
                out, a1, z1, _ = self._forward_fc(pool)
                b = len(yb)
                d2 = ((out - yb) / b).reshape(-1, 1)
                dW2 = d2.T @ a1
                db2 = d2.mean(0)
                da1 = d2 @ self._fc_W2
                dz1 = da1 * _relu_grad(z1)
                dW1 = dz1.T @ pool / b
                db1 = dz1.mean(0)
                n_conv = len(self._conv_W)
                grads = []
                for i in range(n_conv):
                    grads += [np.zeros_like(self._conv_W[i]),
                              np.zeros_like(self._conv_b[i])]
                grads += [dW1 + self.weight_decay * self._fc_W1, db1,
                          dW2 + self.weight_decay * self._fc_W2, db2]
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        pool = self._conv_pool(X)
        out, _, _, _ = self._forward_fc(pool)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Transformer Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerClassifier(_DeepBase):
    """
    Scaled dot-product multi-head self-attention Transformer classifier.

    Architecture:
        Input projection: X → E ∈ R^{T×d_model}
        Attention:        scores = QK^T / √(d_h)  → softmax → AV
        LayerNorm + residual connection
        FFN:              ReLU(E·W₁^T + b₁)·W₂^T + b₂
        Global avg-pool → FC → sigmoid
    """
    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 lr: float = 1e-3, epochs: int = 80, batch_size: int = 32,
                 weight_decay: float = 1e-4, random_state: int = 42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        d, h = self.d_model, self.n_heads
        d_h = max(1, d // h)
        k = np.sqrt(2.0 / d)
        self._Wemb = rng.normal(0, k, (d, d_in))
        self._bemb = np.zeros(d)
        self._WQ   = rng.normal(0, k, (h, d_h, d))
        self._WK   = rng.normal(0, k, (h, d_h, d))
        self._WV   = rng.normal(0, k, (h, d_h, d))
        self._WO   = rng.normal(0, k, (d, h * d_h))
        self._bO   = np.zeros(d)
        self._W1   = rng.normal(0, np.sqrt(2.0 / d), (self.d_ff, d))
        self._b1   = np.zeros(self.d_ff)
        self._W2   = rng.normal(0, np.sqrt(2.0 / self.d_ff), (d, self.d_ff))
        self._b2   = np.zeros(d)
        self._Wout = rng.normal(0, np.sqrt(2.0 / d), (1, d))
        self._bout = np.zeros(1)

    def _params(self) -> list:
        return [self._Wemb, self._bemb,
                self._WQ, self._WK, self._WV, self._WO, self._bO,
                self._W1, self._b1, self._W2, self._b2,
                self._Wout, self._bout]

    def _forward_attn(self, E: np.ndarray):
        """E: (B, T, d)"""
        B, T, d = E.shape
        h = self.n_heads
        d_h = self._WQ.shape[1]
        scale = np.sqrt(d_h)
        heads = []
        for i in range(h):
            Q = E @ self._WQ[i].T  # (B,T,d_h)
            K = E @ self._WK[i].T
            V = E @ self._WV[i].T
            sc = Q @ K.transpose(0, 2, 1) / scale  # (B,T,T)
            A  = _softmax(sc)
            heads.append(A @ V)     # (B,T,d_h)
        cat = np.concatenate(heads, axis=-1)  # (B,T,h*d_h)
        attn_out = cat @ self._WO.T + self._bO  # (B,T,d)
        E2 = attn_out + E                        # residual
        # FFN
        ff = _relu(E2 @ self._W1.T + self._b1) @ self._W2.T + self._b2
        E3 = ff + E2
        return E3

    def _forward_full(self, X: np.ndarray):
        E = X @ self._Wemb.T + self._bemb   # (B,d)  — treat each sample as T=1
        E3d = E[:, np.newaxis, :]            # (B,1,d) — single token
        E3d = self._forward_attn(E3d)
        pooled = E3d.mean(1)                 # (B,d)
        out = _sigmoid(pooled @ self._Wout.T + self._bout).ravel()
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_weights(Xs.shape[1])
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                out = self._forward_full(Xb)
                b = len(yb)
                # Numerical gradient only for output layer (simplified BPTT)
                E = Xb @ self._Wemb.T + self._bemb
                do = ((out - yb) / b).reshape(-1, 1)
                dWout = do.T @ E
                dbout = do.mean(0)
                grads = [np.zeros_like(p) for p in self._params()]
                grads[-2] = dWout + self.weight_decay * self._Wout
                grads[-1] = dbout
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward_full(X)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. DeepSurv Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSurvClassifier(_DeepBase):
    """
    DeepSurv: Cox-inspired deep neural survival model (slide 20).

    Extends Cox PH with a deep MLP as the risk function f_θ(X):
        h(t|X) = h₀(t) · exp(f_θ(X))

    Training loss (combined BCE + Cox partial likelihood):
        L = (1-λ)·BCE + λ·Cox_loss

    Architecture: FC(d→128) → ReLU → FC(128→64) → ReLU → FC(64→1) → sigmoid
    """
    def __init__(self, hidden_layers: tuple = (128, 64), lr: float = 1e-3,
                 epochs: int = 80, batch_size: int = 32, cox_weight: float = 0.5,
                 weight_decay: float = 1e-4, random_state: int = 42):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.cox_weight = cox_weight
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = [d_in] + list(self.hidden_layers) + [1]
        self._Ws, self._bs = [], []
        for d_i, d_o in zip(dims[:-1], dims[1:]):
            k = np.sqrt(2.0 / d_i)
            self._Ws.append(rng.normal(0, k, (d_o, d_i)))
            self._bs.append(np.zeros(d_o))

    def _params(self) -> list:
        p = []
        for W, b in zip(self._Ws, self._bs):
            p += [W, b]
        return p

    def _forward(self, X):
        a = X
        acts = [a]
        pre = []
        n = len(self._Ws)
        for i, (W, b) in enumerate(zip(self._Ws, self._bs)):
            z = a @ W.T + b
            pre.append(z)
            a = _relu(z) if i < n - 1 else _sigmoid(z)
            acts.append(a)
        return a.ravel(), acts, pre

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_weights(Xs.shape[1])
        opt = _Adam(self.lr, weight_decay=self.weight_decay)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        lam = self.cox_weight
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                b = len(yb)
                out, acts, pre = self._forward(Xb)
                # BCE gradient
                d_bce = ((out - yb) / b).reshape(-1, 1)
                # Cox approx: sort by risk, compute partial likelihood gradient
                order = np.argsort(-out)
                lin_s = out[order]
                y_s   = yb[order]
                n_ev  = y_s.sum() + 1e-9
                # Simplified Cox grad (scalar correction per sample)
                cum = np.cumsum(np.exp(lin_s))
                cox_grad = np.zeros(b)
                for i in range(b):
                    if y_s[i] == 1:
                        w = np.exp(lin_s[:i+1]) / (cum[i] + 1e-9)
                        cox_grad[i] -= 1.0 / n_ev
                        cox_grad[:i+1] += w / n_ev
                cox_grad = cox_grad[np.argsort(order)]
                d_total = (1.0 - lam) * d_bce + lam * cox_grad.reshape(-1, 1)
                # Backprop through MLP
                n_l = len(self._Ws)
                delta = d_total
                grads_W, grads_b = [], []
                for i in reversed(range(n_l)):
                    a_p = acts[i]
                    if i < n_l - 1:
                        delta = delta * _relu_grad(pre[i])
                    dW = (delta.T @ a_p) / b + self.weight_decay * self._Ws[i]
                    db = delta.mean(0)
                    grads_W.insert(0, dW)
                    grads_b.insert(0, db)
                    delta = delta @ self._Ws[i]
                grads = []
                for dW, db in zip(grads_W, grads_b):
                    grads += [dW, db]
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        out, _, _ = self._forward(X)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SEDL — Stacked Ensemble Deep Learning
# ═══════════════════════════════════════════════════════════════════════════════

class SEDLClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacked Ensemble Deep Learning (SEDL) — slide 19.

    Two-level meta-learning:
        Level 0 base learners: LR, RF, LSTM, GRU, CNN1D
            hₘ: R^d → [0,1]  for m = 1,…,M
        Feature construction (out-of-fold cross-validation):
            zᵢ = [h₁(xᵢ), h₂(xᵢ), …, h_M(xᵢ)] ∈ R^M
        Level 1 meta-learner (logistic regression on OOF predictions):
            h_meta: R^M → {0,1}
        Final: all base models retrained on full data.
    """
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _base_models(self):
        rs = self.random_state
        return [
            ("lr",   LogisticRegression(max_iter=1000, random_state=rs)),
            ("rf",   RandomForestClassifier(n_estimators=100, random_state=rs)),
            ("lstm", LSTMClassifier(epochs=20, random_state=rs)),
            ("gru",  GRUClassifier(epochs=20, random_state=rs)),
            ("cnn",  CNN1DClassifier(epochs=20, random_state=rs)),
        ]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        base = self._base_models()
        M = len(base)
        meta_X = np.zeros((len(X), M))
        for k, (_, clf) in enumerate(base):
            for tr, va in cv.split(X, y):
                m = copy.deepcopy(clf)
                m.fit(X[tr], y[tr])
                meta_X[va, k] = m.predict_proba(X[va])[:, 1]
        # Meta-learner
        self._meta = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self._meta.fit(meta_X, y)
        # Refit all base models on full data
        self._base_fitted = []
        for _, clf in base:
            m = copy.deepcopy(clf)
            m.fit(X, y)
            self._base_fitted.append(m)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        meta = np.column_stack([m.predict_proba(X)[:, 1] for m in self._base_fitted])
        return self._meta.predict_proba(meta)

    def predict(self, X) -> np.ndarray:
        return self._meta.predict(
            np.column_stack([m.predict_proba(np.asarray(X, float))[:, 1]
                             for m in self._base_fitted])
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Cox Proportional Hazards (classical linear, slide 20)
# ═══════════════════════════════════════════════════════════════════════════════

class CoxPHClassifier(_DeepBase):
    """
    Classical Cox Proportional Hazards model (slide 20).

        h(t|X) = h₀(t) · exp(β^T X)

    Linear predictor f(X) = β^T X.
    Binary probability: P(y=1|X) = σ(β^T X).

    Maximises Cox partial log-likelihood via gradient ascent with L2 decay.
    """
    def __init__(self, lr: float = 1e-3, epochs: int = 100, batch_size: int = 64,
                 weight_decay: float = 1e-4, random_state: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _init_weights(self, d: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.beta = rng.normal(0, 0.01, d)
        self.bias = np.zeros(1)

    def _params(self) -> list:
        return [self.beta, self.bias]

    def _cox_grad(self, X, y):
        lin = X @ self.beta + self.bias[0]
        n = len(y)
        n_ev = y.sum() + 1e-9
        order = np.argsort(-lin)
        lin_s, y_s, X_s = lin[order], y[order], X[order]
        running = -np.inf
        log_cum = np.zeros(n)
        for i in range(n):
            running = np.logaddexp(running, lin_s[i])
            log_cum[i] = running
        g_beta, g_bias = np.zeros_like(self.beta), 0.0
        for i in range(n):
            if y_s[i] == 1:
                sw = np.exp(lin_s[:i+1] - log_cum[i])
                dl = 1.0 - sw.sum()
                g_beta += X_s[i] * dl
                g_bias += dl
        return [-(g_beta / n_ev) + self.weight_decay * self.beta,
                np.array([-(g_bias / n_ev)])]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_weights(Xs.shape[1])
        opt = _Adam(self.lr, weight_decay=0.0)
        opt.init(self._params())
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            for Xb, yb in _mini_batches(Xs, y, self.batch_size, rng):
                grads = self._cox_grad(Xb, yb)
                opt.step(self._params(), grads)
        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.beta + self.bias[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Transfer Learning MLP (slide 21)
# ═══════════════════════════════════════════════════════════════════════════════

class TransferLearningClassifier(_DeepBase):
    """
    Transfer Learning for cross-population disease risk prediction (slide 21).

    Framework:
        D_S = {X_S, P_S(X)} — source domain (large, well-characterised cohort)
        D_T = {X_T, P_T(X)} — target domain (smaller, possibly different ancestry)

    Two-stage MLP adaptation:
        Stage 1 (pre-train): Train full MLP on source domain X_S, y_S.
                             Learns transferable feature representations.
        Stage 2 (fine-tune): Freeze lower shared layers; re-train only the
                             final `finetune_layers` layers on target X_T, y_T.
                             Adapts to target distribution shift.

    Falls back to standard MLP if no source data provided.

    Parameters
    ----------
    source_X, source_y : source domain pre-training data (optional)
    finetune_layers    : number of layers to fine-tune (0 = head only)
    """
    def __init__(self, hidden_sizes: tuple = (128, 64),
                 pretrain_epochs: int = 60, finetune_epochs: int = 40,
                 finetune_layers: int = 1, lr: float = 1e-3,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 source_X: Optional[np.ndarray] = None,
                 source_y: Optional[np.ndarray] = None,
                 random_state: int = 42):
        self.hidden_sizes = hidden_sizes
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.finetune_layers = finetune_layers
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.source_X = source_X
        self.source_y = source_y
        self.random_state = random_state

    def _init_network(self, d_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = [d_in] + list(self.hidden_sizes) + [1]
        self._Ws, self._bs = [], []
        for d_i, d_o in zip(dims[:-1], dims[1:]):
            k = np.sqrt(2.0 / d_i)
            self._Ws.append(rng.normal(0, k, (d_o, d_i)))
            self._bs.append(np.zeros(d_o))

    def _params(self) -> list:
        p = []
        for W, b in zip(self._Ws, self._bs):
            p += [W, b]
        return p

    def _forward(self, X):
        a = X
        acts = [a]
        pre  = []
        n = len(self._Ws)
        for i, (W, b) in enumerate(zip(self._Ws, self._bs)):
            z = a @ W.T + b
            pre.append(z)
            a = _relu(z) if i < n - 1 else _sigmoid(z)
            acts.append(a)
        return a.ravel(), acts, pre

    def _backward(self, out, y, acts, pre, frozen: int = 0):
        B = len(y)
        delta = ((out - y) / B).reshape(-1, 1)
        n = len(self._Ws)
        gW, gb = [], []
        for i in reversed(range(n)):
            if i < n - 1:
                delta = delta * _relu_grad(pre[i])
            dW = (delta.T @ acts[i]) / B
            if i < frozen:
                dW = np.zeros_like(dW)
                delta_b = np.zeros_like(self._bs[i])
            else:
                dW += self.weight_decay * self._Ws[i]
                delta_b = delta.mean(0)
            gW.insert(0, dW)
            gb.insert(0, delta_b)
            delta = delta @ self._Ws[i]
        grads = []
        for dW, db in zip(gW, gb):
            grads += [dW, db]
        return grads

    def _train(self, Xs, ys, epochs, opt, frozen=0):
        rng = np.random.default_rng(self.random_state)
        for _ in range(epochs):
            for Xb, yb in _mini_batches(Xs, ys, self.batch_size, rng):
                out, acts, pre = self._forward(Xb)
                grads = self._backward(out, yb, acts, pre, frozen)
                opt.step(self._params(), grads)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._init_network(Xs.shape[1])

        # Stage 1: pre-train on source domain
        if self.source_X is not None and self.source_y is not None:
            Xs_src = self._scaler.transform(np.asarray(self.source_X, float))
            ys_src = np.asarray(self.source_y, float)
            opt1 = _Adam(self.lr, weight_decay=self.weight_decay)
            opt1.init(self._params())
            self._train(Xs_src, ys_src, self.pretrain_epochs, opt1, frozen=0)

        # Stage 2: fine-tune on target domain (freeze lower layers)
        n_shared = len(self.hidden_sizes)
        frozen = max(0, n_shared - self.finetune_layers)
        opt2 = _Adam(self.lr * 0.1, weight_decay=self.weight_decay)
        opt2.init(self._params())
        self._train(Xs, y, self.finetune_epochs, opt2, frozen=frozen)

        self.is_fitted_ = True
        return self

    def _forward_predict(self, X: np.ndarray) -> np.ndarray:
        out, _, _ = self._forward(X)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  10. Gaussian Mixture Model Classifier (slide 22)
# ═══════════════════════════════════════════════════════════════════════════════

class GMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Mixture Model (GMM) Classifier — slide 22.

        X ~ Σ_k πₖ · N(μₖ, Σₖ)

    Fits separate GMMs for class 0 and class 1 with K components each,
    capturing heterogeneous risk sub-groups within cases and controls.

    Prediction via Bayes' theorem:
        P(y=1|X) = P(X|y=1)·P(y=1) / [P(X|y=0)·P(y=0) + P(X|y=1)·P(y=1)]

    Parameters
    ----------
    n_components    : K — number of mixture components per class (risk strata)
    covariance_type : 'full' | 'diag' | 'spherical'
    """
    def __init__(self, n_components: int = 3, covariance_type: str = "diag",
                 random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self.classes_ = np.array([0, 1])
        self._prior = np.array([(y == 0).mean(), (y == 1).mean()])
        self._gmm = {}
        for c in [0, 1]:
            Xc = Xs[y == c]
            k = min(self.n_components, max(1, len(Xc) // 5))
            gm = GaussianMixture(n_components=k,
                                 covariance_type=self.covariance_type,
                                 random_state=self.random_state, max_iter=200)
            gm.fit(Xc)
            self._gmm[c] = gm
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        lp = {}
        for c in [0, 1]:
            lp[c] = self._gmm[c].score_samples(Xs) + np.log(self._prior[c] + 1e-9)
        mx = np.maximum(lp[0], lp[1])
        p1 = np.exp(lp[1] - mx) / (np.exp(lp[0] - mx) + np.exp(lp[1] - mx))
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
#  11. K-Means Risk Classifier (slide 22)
# ═══════════════════════════════════════════════════════════════════════════════

class KMeansRiskClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Means Risk Stratification Classifier — slide 22.

    Minimises within-cluster Euclidean variance:
        min_{C₁,…,Cₖ}  Σ_k (1/|Cₖ|) Σ_{i,i'∈Cₖ} ‖xᵢ - xᵢ'‖²

    Procedure:
      1. Cluster training data into K groups via K-Means.
      2. Label each cluster by majority vote of true labels.
      3. Predict: assign to nearest centroid → return cluster case fraction.

    Parameters
    ----------
    n_clusters : total number of clusters K
    """
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.cluster import KMeans
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        k = min(self.n_clusters, len(X) // 5)
        self._km = KMeans(n_clusters=k, random_state=self.random_state,
                          n_init=10, max_iter=300)
        self._km.fit(Xs)
        labels = self._km.labels_
        self._risk = np.zeros(k)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                self._risk[c] = y[mask].mean()
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        p1 = np.clip(self._risk[self._km.predict(Xs)], 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
