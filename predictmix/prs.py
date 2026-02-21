"""
prs.py
======
Polygenic Risk Score (PRS) computation.

Functions
---------
compute_prs_from_genotypes
    Standard weighted sum:  PRS_i = Σ_j G_{ij} · β_j

compute_hprs_fusion
    Holistic H-PRS fusion (slide 6 of PredictMix Framework):
        H-PRS_i = π₁·PRS^(LD)  +  π₂·PRS^(L1)  +  π₃·PRS^(annot)

    Three complementary strategies:
      1. LDpred2-style   – Gaussian shrinkage:    β̂_j = ρ·β_j  (ρ = h²/(h²+M(1-h²)/n))
      2. Lassosum2-style – L1 sparsity:           soft-threshold coordinate descent
      3. PRS-CSx-style   – Annotation weighting:  ω_j·β_j  (functional annotation prior)

    Mixing weights π auto-estimated via softmax of per-component signal variances.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from .config import PredictMixConfig


# ─────────────────────────────────────────────────────────────────────────────
#  1. Standard weighted-sum PRS
# ─────────────────────────────────────────────────────────────────────────────

def compute_prs_from_genotypes(
    df: pd.DataFrame,
    cfg: PredictMixConfig,
) -> pd.DataFrame:
    """
    Standard PRS:  PRS_i = Σ_j G_{ij} · β_j

    Requires cfg.genotype_prefix and cfg.beta_file to be set.
    Returns df unchanged if either is None.
    """
    if cfg.genotype_prefix is None or cfg.beta_file is None:
        return df

    betas = pd.read_csv(cfg.beta_file).set_index("snp")["beta"]
    geno_cols = [c for c in df.columns if c.startswith(cfg.genotype_prefix)]
    common = [c for c in geno_cols if c in betas.index]
    if not common:
        raise ValueError(
            f"No overlap between genotype columns (prefix='{cfg.genotype_prefix}') "
            f"and beta_file SNPs."
        )
    df = df.copy()
    df[cfg.prs_column] = df[common].to_numpy(float) @ betas[common].to_numpy(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. H-PRS Component models
# ─────────────────────────────────────────────────────────────────────────────

def _ldpred2_shrinkage(
    G: np.ndarray,
    betas: np.ndarray,
    heritability: float = 0.5,
) -> np.ndarray:
    """
    LDpred2-style Gaussian shrinkage prior (infinitesimal model).

    Under β_j ~ N(0, h²/M):
        ρ = h² / (h² + M(1-h²)/n)
        β̂_j^(LD) = ρ · β_j^(raw)
    """
    n, M = G.shape
    h2 = float(np.clip(heritability, 1e-3, 0.999))
    rho = h2 / (h2 + M * (1.0 - h2) / n)
    return G @ (rho * betas)


def _lassosum2_l1(
    G: np.ndarray,
    betas: np.ndarray,
    lam: float = 0.05,
    n_iter: int = 200,
) -> np.ndarray:
    """
    Lassosum2-style L1 sparsity via coordinate-descent soft-thresholding.

    Solve:  β̂ = argmin_β ½‖y - Gβ‖²/n + λ‖β‖₁
    Using marginal summary statistics as plug-in target.
    Sets weak-effect SNPs exactly to zero (sparse PRS).
    """
    b = betas.copy().astype(float)
    for _ in range(n_iter):
        for j in range(len(b)):
            rj = float(betas[j])
            b[j] = np.sign(rj) * max(abs(rj) - lam, 0.0)
    return G @ b


def _prscsx_annotation(
    G: np.ndarray,
    betas: np.ndarray,
    annotations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    PRS-CSx-style functional annotation weighting.

    ω_j ∈ [0,1] per SNP.
    If no annotations provided, derive from |β_j| magnitude (data-driven).
    β̂_j^(annot) = ω_j · β_j^(raw)
    """
    if annotations is not None:
        omega = np.clip(annotations, 0.0, 1.0)
    else:
        ab = np.abs(betas)
        omega = ab / (ab.max() + 1e-9)
    return G @ (omega * betas)


# ─────────────────────────────────────────────────────────────────────────────
#  3. H-PRS Fusion
# ─────────────────────────────────────────────────────────────────────────────

def compute_hprs_fusion(
    df: pd.DataFrame,
    cfg: PredictMixConfig,
    heritability: float = 0.5,
    lasso_lambda: float = 0.05,
    annotations_col_prefix: Optional[str] = None,
    pi_weights: Optional[tuple] = None,
    output_col: str = "hprs",
) -> pd.DataFrame:
    """
    Holistic PRS (H-PRS) Fusion — PredictMix Framework, slide 6.

    Fuses three PRS strategies:
        H-PRS_i = π₁·PRS^(LD) + π₂·PRS^(L1) + π₃·PRS^(annot)

    Mixing weights π estimated by softmax over per-component variance,
    or fixed via pi_weights=(π₁, π₂, π₃).

    Parameters
    ----------
    df                     : input dataframe with genotype columns
    cfg                    : PredictMixConfig (needs genotype_prefix, beta_file)
    heritability           : trait h² estimate (default 0.5)
    lasso_lambda           : L1 penalty for Lassosum2 component
    annotations_col_prefix : column prefix for per-SNP annotation scores
    pi_weights             : optional fixed mixing weights (π₁, π₂, π₃)
    output_col             : name of added H-PRS column

    Returns
    -------
    df copy with `output_col` containing H-PRS scores.
    """
    if cfg.genotype_prefix is None or cfg.beta_file is None:
        return df

    betas_df  = pd.read_csv(cfg.beta_file).set_index("snp")["beta"]
    geno_cols = [c for c in df.columns if c.startswith(cfg.genotype_prefix)]
    common    = [c for c in geno_cols if c in betas_df.index]
    if not common:
        raise ValueError("No overlap between genotype columns and beta_file SNPs.")

    G = df[common].to_numpy(float)
    b = betas_df[common].to_numpy(float)
    # Standardise genotypes for numerical stability
    G_std = (G - G.mean(0)) / (G.std(0) + 1e-9)

    # Optional per-SNP annotation scores
    annot = None
    if annotations_col_prefix:
        ac = [c for c in df.columns
              if c.startswith(annotations_col_prefix) and c in common]
        if ac:
            annot = df[ac].mean(1).to_numpy()

    # Compute three PRS components
    prs_ld    = _ldpred2_shrinkage(G_std, b, heritability)
    prs_l1    = _lassosum2_l1(G_std, b, lam=lasso_lambda)
    prs_annot = _prscsx_annotation(G_std, b, annot)

    # Auto-estimate mixing weights via softmax of signal variances
    if pi_weights is not None:
        pi = np.asarray(pi_weights, float)
        pi /= pi.sum()
    else:
        v = np.array([np.var(prs_ld), np.var(prs_l1), np.var(prs_annot)]) + 1e-9
        ev = np.exp(v - v.max())
        pi = ev / ev.sum()

    df = df.copy()
    df[output_col] = pi[0] * prs_ld + pi[1] * prs_l1 + pi[2] * prs_annot
    return df
