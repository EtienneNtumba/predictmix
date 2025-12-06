from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .config import PredictMixConfig

def compute_prs_from_genotypes(
    df: pd.DataFrame,
    cfg: PredictMixConfig,
) -> pd.DataFrame:
    """
    Si cfg.genotype_prefix et cfg.beta_file sont fournis :
    - lit les effets beta
    - calcule un score PRS_i = sum_j G_ij * beta_j
    - ajoute une colonne cfg.prs_column au dataframe.
    """
    if cfg.genotype_prefix is None or cfg.beta_file is None:
        return df

    betas = pd.read_csv(cfg.beta_file)
    betas = betas.set_index("snp")["beta"]

    geno_cols = [c for c in df.columns if c.startswith(cfg.genotype_prefix)]
    common = [c for c in geno_cols if c in betas.index]
    if not common:
        raise ValueError("Aucune correspondance entre colonnes génotypes et beta_file")

    G = df[common].to_numpy(dtype=float)
    b = betas[common].to_numpy(dtype=float)

    prs = G @ b
    df = df.copy()
    df[cfg.prs_column] = prs
    return df

