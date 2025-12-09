from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from .config import PredictMixConfig

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def split_xy(df: pd.DataFrame, cfg: PredictMixConfig):
    y = df[cfg.target_column].astype(int)
    X = df.drop(columns=[cfg.target_column] + cfg.drop_columns)
    return X, y



