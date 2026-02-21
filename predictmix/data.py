"""
data.py
=======
Data loading utilities for PredictMix.
Supports CSV and Parquet input formats.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from .config import PredictMixConfig


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet dataset."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_xy(
    df: pd.DataFrame,
    cfg: PredictMixConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and binary target y."""
    y = df[cfg.target_column].astype(int)
    drop = [cfg.target_column] + [c for c in cfg.drop_columns if c in df.columns]
    X = df.drop(columns=drop)
    return X, y
