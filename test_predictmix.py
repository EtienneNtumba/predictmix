"""
test_predictmix.py
==================
Complete test suite for PredictMix v0.2.0

Run:
    python test_predictmix.py                # all tests
    python test_predictmix.py --quick        # skip slow models (~30s)
    python test_predictmix.py --model lstm   # test one specific model
    python test_predictmix.py --fs pearson   # test one FS method
"""
import sys
import warnings
import argparse
import traceback
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import installed package ──────────────────────────────────────────────────
try:
    import predictmix
except ImportError:
    print("ERROR: predictmix is not installed.")
    print("Run: pip install -e .")
    sys.exit(1)

from predictmix.config import PredictMixConfig
from predictmix.models import ModelFactory
from predictmix.feature_selection import select_features
from predictmix.prs import compute_prs_from_genotypes, compute_hprs_fusion

# ── Shared fixtures ───────────────────────────────────────────────────────────
RNG  = np.random.default_rng(42)
X_df = pd.DataFrame(RNG.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
y_s  = pd.Series((RNG.normal(size=200) > 0).astype(int))
yn   = y_s.values.astype(float)

DEEP = {"rnn", "lstm", "gru", "cnn1d", "transformer", "coxph", "deepsurv"}

SLOW_MODELS = {"sedl", "transformer", "svm", "ensemble"}

ALL_MODELS = [
    "logreg", "bayesian", "svm", "rf", "mlp",
    "adaboost", "bagging", "ensemble",
    "coxph", "deepsurv",
    "gmm", "kmeans_risk",
    "rnn", "lstm", "gru", "cnn1d", "transformer",
    "sedl",
]

ALL_FS = [
    "none", "pearson", "chi2", "infogain",
    "lasso", "ridge", "elasticnet", "tree",
    "pca", "rfe",
    "dnn_l1", "gated", "autoencoder", "stability", "stacked_dl",
]

QUICK_MODELS = [m for m in ALL_MODELS if m not in SLOW_MODELS]
QUICK_FS     = [fs for fs in ALL_FS if fs not in {"stacked_dl"}]

# ── Test runner ───────────────────────────────────────────────────────────────
PASS    = "\033[92m PASS\033[0m"
FAIL    = "\033[91m FAIL\033[0m"
results = []

def test(name, fn):
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  [{PASS}] {name:<50s} ({elapsed:.1f}s)")
        results.append((name, True, None))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{FAIL}] {name:<50s} ({elapsed:.1f}s)")
        print(f"           {traceback.format_exc().strip().splitlines()[-1]}")
        results.append((name, False, str(e)))


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Config
# ═══════════════════════════════════════════════════════════════════════════════

def test_config():
    print("\n── 1. Config ───────────────────────────────────────────────────")

    def t_defaults():
        cfg = PredictMixConfig()
        assert cfg.model == "ensemble"
        assert cfg.feature_selection == "lasso"
        assert cfg.cv_folds == 5
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42

    def t_custom():
        cfg = PredictMixConfig(model="lstm", feature_selection="gated", n_features=20)
        assert cfg.model == "lstm"
        assert cfg.n_features == 20

    test("Default values", t_defaults)
    test("Custom values", t_custom)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Models
# ═══════════════════════════════════════════════════════════════════════════════

def test_models(model_list):
    print(f"\n── 2. Models ({len(model_list)}) ────────────────────────────────────")
    for m in model_list:
        def _t(m=m):
            clf = ModelFactory(PredictMixConfig(model=m)).build()
            clf.fit(X_df, yn if m in DEEP else y_s)
            p = clf.predict_proba(X_df)
            assert p.shape == (200, 2), f"Expected shape (200,2), got {p.shape}"
            assert 0 <= p[:, 1].min() and p[:, 1].max() <= 1, "Probabilities out of [0,1]"
            pred = clf.predict(X_df)
            assert set(pred).issubset({0, 1}), "predict() returned non-binary values"
        test(f"model={m}", _t)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Feature selection
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_selection(fs_list):
    print(f"\n── 3. Feature Selection ({len(fs_list)}) ─────────────────────────────")
    for fs in fs_list:
        def _t(fs=fs):
            cfg  = PredictMixConfig(feature_selection=fs, n_features=5)
            Xsel, cols = select_features(X_df, y_s, cfg)
            assert Xsel.shape[0] == 200, "Row count changed after feature selection"
            assert len(cols) > 0, "No features selected"
            assert Xsel.shape[1] == len(cols), "Column/names length mismatch"
            assert not np.isnan(Xsel.values).any(), "NaN values in selected features"
        test(f"feature_selection={fs}", _t)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. PRS
# ═══════════════════════════════════════════════════════════════════════════════

def test_prs():
    print("\n── 4. PRS ──────────────────────────────────────────────────────")

    def t_no_genotypes():
        cfg = PredictMixConfig()
        df2 = compute_prs_from_genotypes(X_df.copy(), cfg)
        assert df2.shape == X_df.shape, "DataFrame changed when no PRS config provided"

    def t_hprs_no_genotypes():
        cfg = PredictMixConfig()
        df2 = compute_hprs_fusion(X_df.copy(), cfg)
        assert df2.shape == X_df.shape, "DataFrame changed when no PRS config provided"

    test("Standard PRS — passthrough (no genotype config)", t_no_genotypes)
    test("H-PRS fusion — passthrough (no genotype config)", t_hprs_no_genotypes)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Full pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline():
    import tempfile, os
    from predictmix.pipeline import PredictMixPipeline

    print("\n── 5. Pipeline end-to-end ──────────────────────────────────────")

    def t_train_predict():
        with tempfile.TemporaryDirectory() as tmp:
            # Write dataset to CSV
            data = X_df.copy()
            data["y"] = y_s.values
            data_path = os.path.join(tmp, "train.csv")
            data.to_csv(data_path, index=False)

            cfg = PredictMixConfig(
                model="rf",
                feature_selection="pearson",
                n_features=5,
                output_dir=os.path.join(tmp, "model"),
                cv_folds=3,
            )
            pipe = PredictMixPipeline(cfg)
            metrics = pipe.fit(data_path,
                               export_predictions=os.path.join(tmp, "preds.csv"))

            # Metrics structure
            assert "cv" in metrics and "test" in metrics
            for split in ("cv", "test"):
                for k in ("auc", "accuracy", "f1_macro"):
                    assert k in metrics[split], f"Missing '{k}' in {split} metrics"
                    assert 0.0 <= metrics[split][k] <= 1.0, \
                        f"{split}.{k} = {metrics[split][k]} out of [0,1]"

            # Save / load / predict
            pipe.save()
            model_file = os.path.join(tmp, "model", "predictmix_model.joblib")
            assert os.path.exists(model_file), "Model file not created"

            pipe2  = PredictMixPipeline.load(model_file)
            probas = pipe2.predict_proba(X_df.copy())
            assert len(probas) == 200, "Wrong number of predictions"
            assert probas.min() >= 0 and probas.max() <= 1, "Probas out of [0,1]"

    def t_predictions_csv():
        with tempfile.TemporaryDirectory() as tmp:
            data = X_df.copy()
            data["y"] = y_s.values
            data_path = os.path.join(tmp, "train.csv")
            data.to_csv(data_path, index=False)
            pred_path = os.path.join(tmp, "preds.csv")

            cfg = PredictMixConfig(
                model="logreg",
                feature_selection="none",
                output_dir=tmp,
                cv_folds=3,
            )
            PredictMixPipeline(cfg).fit(data_path, export_predictions=pred_path)

            preds = pd.read_csv(pred_path)
            for col in ("y_true", "risk_proba", "split"):
                assert col in preds.columns, f"Missing column '{col}' in predictions CSV"

    def t_metrics_range():
        with tempfile.TemporaryDirectory() as tmp:
            data = X_df.copy()
            data["y"] = y_s.values
            data.to_csv(os.path.join(tmp, "d.csv"), index=False)
            cfg = PredictMixConfig(model="bayesian", feature_selection="none",
                                   output_dir=tmp, cv_folds=3)
            m = PredictMixPipeline(cfg).fit(os.path.join(tmp, "d.csv"))
            for split in ("cv", "test"):
                auc = m[split]["auc"]
                assert 0.0 <= auc <= 1.0, f"AUC {auc} out of range"

    test("Train → evaluate → save → load → predict", t_train_predict)
    test("Predictions CSV has required columns",      t_predictions_csv)
    test("Metrics are in [0, 1] range",               t_metrics_range)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Risk categories
# ═══════════════════════════════════════════════════════════════════════════════

def test_risk_categories():
    print("\n── 6. Risk Categories ──────────────────────────────────────────")

    def _cat(p):
        if p > 0.80: return "Very High"
        if p > 0.60: return "High"
        if p > 0.40: return "Moderate"
        if p > 0.20: return "Low"
        return "Very Low"

    def t_thresholds():
        cases = [
            (0.05, "Very Low"),
            (0.25, "Low"),
            (0.50, "Moderate"),
            (0.70, "High"),
            (0.90, "Very High"),
        ]
        for p, expected in cases:
            got = _cat(p)
            assert got == expected, f"p={p}: expected '{expected}', got '{got}'"

    def t_boundaries():
        assert _cat(0.20) == "Very Low"
        assert _cat(0.21) == "Low"
        assert _cat(0.40) == "Low"
        assert _cat(0.41) == "Moderate"
        assert _cat(0.60) == "Moderate"
        assert _cat(0.61) == "High"
        assert _cat(0.80) == "High"
        assert _cat(0.81) == "Very High"

    test("Risk category thresholds", t_thresholds)
    test("Risk category boundaries", t_boundaries)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def test_reproducibility():
    print("\n── 7. Reproducibility ──────────────────────────────────────────")

    def t_same_seed():
        clf1 = ModelFactory(PredictMixConfig(model="rf", random_state=42)).build()
        clf2 = ModelFactory(PredictMixConfig(model="rf", random_state=42)).build()
        clf1.fit(X_df, y_s)
        clf2.fit(X_df, y_s)
        p1 = clf1.predict_proba(X_df)[:, 1]
        p2 = clf2.predict_proba(X_df)[:, 1]
        assert np.allclose(p1, p2), "Same seed should give identical predictions"

    def t_different_seed():
        clf1 = ModelFactory(PredictMixConfig(model="rf", random_state=0)).build()
        clf2 = ModelFactory(PredictMixConfig(model="rf", random_state=99)).build()
        clf1.fit(X_df, y_s)
        clf2.fit(X_df, y_s)
        p1 = clf1.predict_proba(X_df)[:, 1]
        p2 = clf2.predict_proba(X_df)[:, 1]
        assert not np.allclose(p1, p2), "Different seeds should give different predictions"

    test("Same seed → identical predictions",    t_same_seed)
    test("Different seeds → different predictions", t_different_seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PredictMix v0.2.0 test suite")
    parser.add_argument("--quick",  action="store_true",
                        help="Skip slow models (svm, ensemble, transformer, sedl)")
    parser.add_argument("--model",  default=None,
                        help="Test one model only  e.g. --model lstm")
    parser.add_argument("--fs",     default=None,
                        help="Test one FS method only  e.g. --fs pearson")
    args = parser.parse_args()

    model_list = ([args.model]  if args.model else
                  QUICK_MODELS  if args.quick else ALL_MODELS)
    fs_list    = ([args.fs]     if args.fs    else
                  QUICK_FS      if args.quick else ALL_FS)

    print("=" * 60)
    print("  PredictMix v0.2.0 — Test Suite")
    print("=" * 60)

    test_config()
    test_models(model_list)
    test_feature_selection(fs_list)
    test_prs()
    test_pipeline()
    test_risk_categories()
    test_reproducibility()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)

    print()
    print("=" * 60)
    print(f"  Results: {n_pass} passed, {n_fail} failed  ({len(results)} total)")
    print("=" * 60)

    if n_fail:
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  ✗ {name}")
                print(f"    {err}")
        sys.exit(1)
    else:
        print("\n  ✅  All tests passed!")


if __name__ == "__main__":
    main()
