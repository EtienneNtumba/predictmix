# PredictMix

[![PyPI version](https://badge.fury.io/py/predictmix.svg)](https://badge.fury.io/py/predictmix)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/EtienneNtumba/predictmix)

### Integrated Polygenic + Clinical Disease Risk Prediction Pipeline

**Developed by:**
- **Etienne Ntumba Kabongo**, McGill University ([etienne.kabongo@mcgill.ca](mailto:etienne.kabongo@mcgill.ca))
- **Prof. Emile R. Chimusa**, Northumbria University ([emile.chimusa@northumbria.ac.uk](mailto:emile.chimusa@northumbria.ac.uk))

---

## Overview

**PredictMix** is a modular, extensible, and production-ready machine-learning pipeline for **integrated disease risk prediction**, combining:

- 🧬 **Polygenic Risk Scores (PRS)** — Bayesian hierarchical model + H-PRS fusion (LDpred2 + Lassosum2 + PRS-CSx)
- 🏥 **Clinical variables**, family history, lifestyle and environmental factors
- 🌍 **Multi-ethnic cohort support** with population-specific calibration
- 🔍 **15 feature selection algorithms** — filter, embedded, wrapper, and deep learning
- 🤖 **18 prediction models** — classical, survival, stratification, transfer learning, and deep learning
- 💡 **LIME Explainability** — local feature importance for any model
- 📊 **Publication-grade visualizations**

Originally developed for **sickle cell disease** and **population stratification in African cohorts** (UK Biobank, H3Africa), PredictMix is fully generalizable to any **binary disease risk prediction** task.

> **New in v0.2.0** — Full implementation of the PredictMix theoretical framework:
> 18 models including CoxPH, GMM, KMeans Risk, Transfer Learning;
> 15 feature selection methods including Pearson correlation, Information Gain, Ridge, RFE;
> H-PRS Fusion model (LDpred2 + Lassosum2 + PRS-CSx).
> All deep learning models run with **NumPy only** — no PyTorch or TensorFlow required.

---

## 🚀 Quick Start

### Installation

```bash
pip install predictmix           # from PyPI (recommended)

git clone https://github.com/EtienneNtumba/predictmix.git
cd predictmix && pip install -e .  # from source
```

### Basic Usage

```bash
# Train with default ensemble model
predictmix train data.csv --target-column disease_status --output-dir my_model

# Train with LSTM + deep feature selection
predictmix train data.csv \
  --target-column disease_status \
  --model lstm \
  --feature-selection dnn_l1 \
  --n-features 10 \
  --output-dir my_lstm_model

# Survival analysis with Cox PH
predictmix train data.csv \
  --target-column disease_status \
  --model coxph \
  --feature-selection pearson \
  --output-dir cox_model

# Risk stratification with GMM
predictmix train data.csv \
  --target-column disease_status \
  --model gmm \
  --feature-selection infogain \
  --output-dir gmm_model

# Transfer learning (cross-population)
predictmix train target_cohort.csv \
  --target-column disease_status \
  --model transfer \
  --output-dir transfer_model

# Benchmark all models
predictmix benchmark data.csv --target-column disease_status

# Predict on new patients
predictmix predict my_model/predictmix_model.joblib new_patients.csv \
  --output predictions.csv

# Generate all plots
predictmix plot my_model/predictions.csv --kind all --output-dir plots/
```

---

## 📊 Test Dataset

**Download:** [test_predictmix_data.csv](https://github.com/EtienneNtumba/predictmix/releases/download/v0.1.1/test_predictmix_data.csv)

- 300 samples (91 cases, 209 controls)
- 13 features: PRS, age, sex, BMI, hemoglobin, clinical markers, SNPs
- Expected AUC: 0.75–0.85, runtime < 2 minutes

---

## 🤖 Prediction Models (18 total)

### Classical & Statistical Models

| Key | Model | Description |
|-----|-------|-------------|
| `logreg` | Logistic Regression | Log-odds: log(ρ/1-ρ) = γ₀ + γ·PRS + γ·Age + γ·F^(r) |
| `bayesian` | Bayesian Classifier | Posterior P(y=1\|X) via Bayes' theorem (Gaussian NB) |
| `svm` | Support Vector Machine | Maximum-margin hyperplane in high-dimensional space |
| `rf` | Random Forest | 500 decision trees; Gini feature importance |
| `mlp` | Multi-Layer Perceptron | sklearn MLP with early stopping (128→64→32) |
| `adaboost` | AdaBoost | Sequential boosting: w = sign(Σ_t αₜ hₜ(x)) |
| `bagging` | Bagging | Bootstrap aggregation: w = argmax Σ_k 1(y = hₖ(x)) |
| `ensemble` | Stacking Ensemble | LR + SVM + RF + AdaBoost → meta LogisticRegression |

### Survival Analysis

| Key | Model | Equation | Description |
|-----|-------|----------|-------------|
| `coxph` | Cox Proportional Hazards | h(t\|X) = h₀(t)·exp(β^T X) | Classical time-to-event; interpretable coefficients |
| `deepsurv` | DeepSurv | h(t\|X) = h₀(t)·exp(f_θ(X)) | Neural network replaces linear predictor |

### Risk Stratification

| Key | Model | Equation | Description |
|-----|-------|----------|-------------|
| `gmm` | Gaussian Mixture Model | X ~ Σ_k πₖ N(μₖ, Σₖ) | Identifies K latent risk strata per class |
| `kmeans_risk` | K-Means Risk | min Σ_k(1/\|Cₖ\|) Σᵢ,ᵢ'∈Cₖ‖xᵢ-xᵢ'‖² | Partitions into K clusters; majority-vote labelling |

### Transfer Learning

| Key | Model | Framework | Description |
|-----|-------|-----------|-------------|
| `transfer` | Transfer Learning MLP | D_S={X_S,P_S(X)} → D_T={X_T,P_T(X)} | Pre-train on source cohort; fine-tune on target population |

### Deep Learning (pure NumPy — no extra deps)

| Key | Model | Architecture | Reference |
|-----|-------|--------------|-----------|
| `rnn` | Recurrent Neural Network | sₜ = f(U·xₜ + W·sₜ₋₁) | Slide 17 |
| `lstm` | Long Short-Term Memory | fₜ,iₜ,oₜ gates + cell state cₜ | Slides 17–18 |
| `gru` | Gated Recurrent Unit | Update/reset gates + candidate state | Slide 16 |
| `cnn1d` | 1-D CNN | cᵢ = f(w^T x_{i:i+h-1} + b) | Slide 18 |
| `transformer` | Transformer | Multi-head self-attention + FFN | Slide 16 |
| `sedl` | Stacked Ensemble DL | Level-0: LR+RF+LSTM+GRU+CNN1D → Level-1: meta-LR | Slide 19 |

---

## 🔍 Feature Selection Methods (15 total)

### Filter Methods — evaluate each feature independently

| Key | Method | Equation |
|-----|--------|----------|
| `none` | Pass-through | All features used |
| `pearson` | Pearson Correlation | r(X,y) = Σ(Xᵢ-X̄)(yᵢ-ȳ) / √[Σ(Xᵢ-X̄)²·Σ(yᵢ-ȳ)²] |
| `chi2` | Chi-Squared Test | χ² = Σ(Oᵢ-Eᵢ)²/Eᵢ — for discrete/non-negative features |
| `infogain` | Information Gain | IG(X_j, y) = H(y) - H(y\|X_j) — mutual information |

### Embedded Methods — regularisation during model training

| Key | Method | Equation |
|-----|--------|----------|
| `lasso` | LASSO (L1) | Cost = (1/n)Σ(yᵢ-ŷᵢ)² + λ Σ_j \|w_j\| — drives weak features to zero |
| `ridge` | Ridge (L2) | Cost = (1/n)Σ(yᵢ-ŷᵢ)² + λ Σ_j w_j² — shrinks coefficients |
| `elasticnet` | Elastic Net | Cost = (1/n)Σ(yᵢ-ŷᵢ)² + λ[(1-α)\|w\|₁ + α\|w\|₂²] — L1+L2 |
| `tree` | Random Forest Importance | Gini-based importance scores |

### Dimensionality Reduction

| Key | Method | Equation |
|-----|--------|----------|
| `pca` | PCA | maximise φ₁₁,…,φₚ₁ {(1/n)Σ(Σⱼ φⱼ₁ xᵢⱼ)²} subject to Σφ²ⱼ₁=1 |

### Wrapper Methods — greedy search

| Key | Method | Description |
|-----|--------|-------------|
| `rfe` | Recursive Feature Elimination | Repeatedly removes least important feature (backward elimination) |

### Deep Learning Feature Selection

| Key | Method | Core equation |
|-----|--------|---------------|
| `dnn_l1` | L1-Regularised DNN | I_j = Σᵢ \|W^(1)_ij\| — sparse input layer |
| `gated` | Gated DFS | X̃ = X ⊙ m ; L = L_task + λ₁‖m‖₁ + λ₂‖W‖² |
| `autoencoder` | Sparse Autoencoder | h = ReLU(WₑX + bₑ) ; L = ‖X-X̂‖² + β‖h‖₁ |
| `stability` | Stability Selection | π_j = (1/B)Σ 1(x_j ∈ X_b) ≥ 0.6 |
| `stacked_dl` | Stacked DL Pipeline | X → L1-DNN → X^(1) → Gated → X^(2) → Sparse AE → h |

---

## 🧬 Advanced PRS Fusion (H-PRS)

PredictMix v0.2.0 implements the full **Holistic PRS (H-PRS)** fusion model (slide 6):

```
H-PRS_i = π₁ · PRS^(LD)_i  +  π₂ · PRS^(L1)_i  +  π₃ · PRS^(annot)_i
```

Three complementary strategies are fused:

| Component | Strategy | Description |
|-----------|----------|-------------|
| PRS^(LD) | LDpred2 | Gaussian shrinkage prior — ρ = h²/(h² + M(1-h²)/n) |
| PRS^(L1) | Lassosum2 | L1 coordinate descent — soft-threshold sparsity |
| PRS^(annot) | PRS-CSx | Functional annotation weighting — ω_j proportional to \|β_j\| |

Mixing weights π are auto-estimated via softmax of per-component signal variances.

```python
from predictmix.prs import compute_hprs_fusion
df = compute_hprs_fusion(df, cfg, heritability=0.5, output_col="hprs")
```

---

## 📖 Command Reference

### `predictmix train`

```bash
predictmix train data.csv [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model, -m` | `ensemble` | Model key (see table above) |
| `--feature-selection, -f` | `lasso` | Feature selection method |
| `--n-features, -k` | `100` | Max features to retain |
| `--target-column, -y` | `y` | Binary target column (0/1) |
| `--output-dir, -o` | `predictmix_output` | Output directory |
| `--cv-folds` | `5` | Cross-validation folds |
| `--plots / --no-plots` | `--no-plots` | Auto-generate ROC + PR curves |
| `--config, -c` | — | YAML config (overrides all flags) |

**Outputs:**
```
my_model/
├── predictmix_model.joblib   # Trained pipeline (reusable)
├── config.json               # Configuration used
├── metrics.json              # CV + test metrics
└── predictions.csv           # Per-sample risk scores
```

---

### `predictmix predict`

```bash
predictmix predict my_model/predictmix_model.joblib new_patients.csv \
  --output predictions.csv --threshold 0.5
```

Three columns added to output:

| Column | Description |
|--------|-------------|
| `risk_proba` | Predicted probability (0–1) |
| `risk_label` | Binary prediction at threshold |
| `risk_category` | Very Low / Low / Moderate / High / Very High |

### `predictmix benchmark`

Train and compare multiple models in one command:

```bash
predictmix benchmark data.csv \
  --target-column disease_status \
  --models "logreg,rf,ensemble,coxph,gmm,lstm,gru,transformer,deepsurv,sedl" \
  --output-dir benchmark_results
```

Outputs `benchmark_summary.csv` with AUC, Accuracy, F1, Precision, Recall per model.

### `predictmix plot`

```bash
predictmix plot predictions.csv --kind all --output-dir plots/
```

`--kind`: `rocpr` | `hist` | `scatter` | `heatmap` | `calib` | `all` (7 plots)

---

## 📐 Mathematical Framework

### Four-Stage Pipeline

```
Step 1: Polygenic Risk Scores
   PRS_i = Σⱼ Σₖ θⱼₖ^(n) γⱼₖ  +  Σⱼ Σₖ (φⱼₖ + F·ωⱼₖ)θⱼₖ^(n) Xⱼ^(n) βⱼₖ  +  Σₖ μₖ αₖ
        ↓
Step 2: Multi-modal predictor fusion
   X = [PRS, Family History, Clinical, Lifestyle, Demographics, Environment]
        ↓
Step 3: Logistic regression baseline
   log(ρ/1-ρ) = γ₀ + γ^(n)·P^(n) + γ₃·P_sib + γ₄·N# + γ₅·PRS_i + γ₆·Age + γ^(r)·F^(r)
        ↓
Step 4: Stratified risk output
   Risk category ∈ {Very Low, Low, Moderate, High, Very High}
```

### Risk Stratification

| Risk Score | Category | Clinical Action |
|------------|----------|----------------|
| > 0.80 | Very High | Aggressive treatment, specialist referral |
| 0.60–0.80 | High | Enhanced surveillance, preventive measures |
| 0.40–0.60 | Moderate | Regular follow-up |
| 0.20–0.40 | Low | Standard screening, lifestyle counselling |
| < 0.20 | Very Low | Routine care |

### AUC Interpretation (slide 25)

| AUC | Interpretation |
|-----|---------------|
| 90–100% | Excellent — outstanding discrimination |
| 80–90% | Good — clinically useful |
| 70–80% | Fair — acceptable, may need refinement |
| 50–70% | Poor — limited predictive value |

---

## 💡 Model Explainability (LIME)

PredictMix integrates LIME (Local Interpretable Model-Agnostic Explanations, slide 26):

```
arg min_{g∈G} L(f, g, πₓ) + Ω(g)
```

- L measures how well the local surrogate g approximates f
- πₓ defines the neighbourhood kernel around instance x
- Ω penalises model complexity

```python
from predictmix.plots import explain_lime
explain_lime(model, X_test, feature_names=["prs","age","bmi",...])
```

---

## 🗂️ Model Selection Guide

| Use case | Recommended model | Feature selection |
|----------|-------------------|------------------|
| Interpretable baseline | `logreg` | `lasso` or `pearson` |
| Best classical performance | `ensemble` | `lasso` or `elasticnet` |
| Time-to-event / survival | `coxph` or `deepsurv` | `pearson` or `dnn_l1` |
| Latent risk groups | `gmm` | `infogain` |
| Unsupervised clustering | `kmeans_risk` | `pca` |
| Cross-population transfer | `transfer` | `stability` |
| Temporal / sequential data | `lstm` or `gru` | `gated` |
| High-dimensional genomics | `rf` or `ensemble` | `stability` |
| Novel deep representation | `transformer` | `stacked_dl` |
| Maximum ensemble power | `sedl` | `stacked_dl` |
| Model comparison | `benchmark` command | any |

---

## 📦 Dependencies

```
numpy  pandas  scikit-learn>=1.0  scipy  joblib  pyyaml  typer  matplotlib  lime
```

All 8 deep learning models are implemented in **pure NumPy** — no PyTorch, TensorFlow, or JAX required.

---

## 🏗️ Project Structure

```
predictmix/
├── __init__.py            # v0.2.0
├── config.py              # PredictMixConfig (18 models, 15 FS methods)
├── data.py                # CSV / Parquet data loading
├── prs.py                 # PRS + H-PRS fusion (LDpred2 + Lassosum2 + PRS-CSx)
├── feature_selection.py   # 15 feature selection methods
├── deep_models.py         # 8 DL models + CoxPH + TransferLearning + GMM + KMeans
├── models.py              # ModelFactory — 18 model builders
├── pipeline.py            # Training, CV, evaluation, save/load, predict
├── plots.py               # ROC, PR, calibration, confusion, scatter, volcano, LIME
└── cli.py                 # CLI: train | predict | plot | benchmark | plot-volcano
```

---

## 🔄 Changelog

### v0.2.0 (2025)
**New models:**
- `coxph` — Classical Cox Proportional Hazards: h(t|X) = h₀(t)·exp(β^T X)
- `gmm` — Gaussian Mixture Model: X ~ Σ_k πₖ N(μₖ, Σₖ)
- `kmeans_risk` — K-Means risk stratification classifier
- `transfer` — Transfer Learning MLP (pre-train/fine-tune)
- `rnn`, `lstm`, `gru`, `cnn1d`, `transformer`, `deepsurv`, `sedl` (deep learning)
- `bayesian`, `adaboost`, `bagging` (classical)

**New feature selection:**
- `pearson` — Pearson correlation filter
- `infogain` — Information Gain / mutual information filter
- `ridge` — Ridge L2 embedded method
- `rfe` — Recursive Feature Elimination (wrapper)
- `dnn_l1`, `gated`, `autoencoder`, `stability`, `stacked_dl` (deep learning)

**New PRS:**
- H-PRS fusion: LDpred2 + Lassosum2 + PRS-CSx (`compute_hprs_fusion`)

**New CLI:**
- `benchmark` command — compare all models in one run
- `predict` now outputs `risk_category` column
- All DL models run in pure NumPy — zero new dependencies

### v0.1.1 (2024)
- Initial public release

---

## 📚 Citation

```bibtex
@software{predictmix2025,
  author    = {Ntumba Kabongo, Etienne and Chimusa, Emile R.},
  title     = {PredictMix: An Integrated Polygenic-Clinical Machine Learning
               Pipeline for Disease Risk Prediction},
  year      = {2025},
  version   = {0.2.0},
  url       = {https://github.com/EtienneNtumba/predictmix}
}
```

---

## 🗺️ Roadmap

- [ ] SHAP explainability integration
- [ ] Multi-class classification support
- [ ] Integration with PRS-CS and LDpred2 (direct PLINK interface)
- [ ] Automated genotype QC and population stratification
- [ ] Nextflow / Snakemake HPC workflows
- [ ] Interactive risk dashboards for clinicians
- [ ] PyTorch / JAX GPU backend

---

## 🔗 Related Tools

- [PRSice-2](https://github.com/choishingwan/PRSice) — PRS calculation
- [LDpred2 / bigsnpr](https://github.com/privefl/bigsnpr) — Bayesian PRS with LD
- [PRS-CSx](https://github.com/getian107/PRScsx) — Cross-ancestry PRS
- [PLINK2](https://www.cog-genomics.org/plink/2.0/) — Genetic data analysis
- [LIME](https://github.com/marcotcr/lime) — Model explainability

---

## 📄 License

MIT License. Developed at **McGill University** and **Northumbria University**.

Contact: [etienne.kabongo@mcgill.ca](mailto:etienne.kabongo@mcgill.ca)

---

*Made with ❤️ for the genomics and precision medicine community*
