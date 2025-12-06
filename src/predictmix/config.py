from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from typing_extensions import Literal

ModelName = Literal["logreg", "svm", "rf", "mlp", "ensemble"]
FSMethod = Literal["none", "lasso", "elasticnet", "tree", "chi2", "pca"]


@dataclass
class PredictMixConfig:
    target_column: str = "y"
    id_column: Optional[str] = None

    prs_column: Optional[str] = "prs"
    genotype_prefix: Optional[str] = None
    beta_file: Optional[str] = None

    feature_selection: FSMethod = "lasso"
    n_features: Optional[int] = 100

    model: ModelName = "ensemble"
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.2

    output_dir: str = "predictmix_output"

    drop_columns: List[str] = field(default_factory=list)

