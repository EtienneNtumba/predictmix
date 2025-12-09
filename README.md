# PredictMix

[![PyPI version](https://badge.fury.io/py/predictmix.svg)](https://badge.fury.io/py/predictmix)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Integrated Polygenic + Clinical Disease Risk Prediction Pipeline

**Developed by:**
- **Etienne Ntumba Kabongo**, McGill University ([etienne.kabongo@mcgill.ca](mailto:etienne.kabongo@mcgill.ca))
- **Prof. Emile R. Chimusa**, Northumbria University ([emile.chimusa@northumbria.ac.uk](mailto:emile.chimusa@northumbria.ac.uk))

---

## Overview

**PredictMix** is a modular and extensible machine-learning pipeline for **integrated disease risk prediction**, combining:

- 🧬 **Polygenic Risk Scores (PRS)**
- 🏥 **Clinical variables**
- 🌍 **Environmental and lifestyle factors**
- 🔍 **Feature selection algorithms**
- 🤖 **Multiple ML models**
- 💡 **Explainability** (LIME-ready architecture)
- 📊 **Publication-grade visualizations**

Originally developed for genomic studies on **sickle cell disease** and **population stratification in African cohorts**, PredictMix is fully generalizable to any **binary disease risk prediction** task.

**Ideal for:**
- Researchers in statistical genetics, epidemiology, and AI-driven clinical modeling
- Large-scale biobank analyses (e.g., UK Biobank, China Kadoorie Biobank, H3Africa)
- Rare disease prediction and stratification
- Integrative genomic & clinical prediction studies

---

## 🚀 Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install predictmix

# From conda-forge (coming soon!)
# conda install -c conda-forge predictmix

# From source
git clone https://github.com/EtienneNtumba/predictmix.git
cd predictmix
pip install -e .
```

### Basic Usage

```bash
# Train a model
predictmix train data.csv \
  --target-column disease_status \
  --model ensemble \
  --feature-selection lasso \
  --n-features 10 \
  --output-dir my_model

# Generate visualizations
predictmix plot my_model/predictions.csv \
  --kind all \
  --output-dir my_plots

# Make predictions on new data
predictmix predict my_model/predictmix_model.joblib new_patients.csv \
  --output predictions.csv
```

---

## 📊 Test Dataset

We provide a synthetic test dataset to help you get started:

**Download:** [test_predictmix_data.csv](https://github.com/EtienneNtumba/predictmix/releases/download/v0.1.1/test_predictmix_data.csv)

**Dataset specifications:**
- 300 samples (91 cases, 209 controls)
- 13 features: PRS, age, sex, BMI, hemoglobin, clinical markers, SNPs
- Realistic associations between features and disease outcome

### Try it now:

```bash
# Download the test dataset
wget https://github.com/EtienneNtumba/predictmix/releases/download/v0.1.1/test_predictmix_data.csv

# Train and evaluate
predictmix train test_predictmix_data.csv \
  --target-column disease_status \
  --model ensemble \
  --feature-selection lasso \
  --n-features 8 \
  --output-dir test_results 

# Generate all plots
predictmix plot test_results/predictions.csv \
  --kind all \
  --output-dir test_plots
```

**Expected results:**
- AUC: 0.75-0.85
- Accuracy: 70-80%
- 7 publication-ready plots generated

---

## Key Features

### 🔬 End-to-End Prediction Pipeline
- Automated train/test split with stratification
- Cross-validation (configurable)
- Multiple ML models: logistic regression, SVM, Random Forest, MLP, ensemble

### 🧬 Multi-modal Feature Integration
- Seamlessly combine PRS + clinical + environmental + biochemical data
- Flexible column configuration
- Handle mixed data types

### 🔍 Feature Selection Methods
- `none` - Use all features
- `lasso` - L1 regularization
- `elasticnet` - L1 + L2 regularization
- `tree` - Random Forest importance
- `chi2` - Chi-squared test
- `pca` - Principal Component Analysis

### 📊 Advanced Visualization Suite

Generate publication-quality plots:

| Plot Type | Description |
|-----------|-------------|
| ROC Curve | Receiver Operating Characteristic |
| PR Curve | Precision-Recall curve |
| Calibration | Model calibration assessment |
| Confusion Matrix | Classification performance heatmap |
| Risk Histograms | Distribution of predicted risk scores |
| Scatter Plots | Risk vs. true class visualization |
| Volcano Plot | GWAS summary statistics visualization |

### 📦 Easy Installation & CLI-first Design

```bash
pip install predictmix
predictmix --help
```

---

## Documentation

### Command Overview

```bash
predictmix --help
```

Available commands:
- `train` - Train a PredictMix model
- `predict` - Apply trained model to new data
- `plot` - Generate visualization plots
- `plot-volcano` - Create volcano plots for GWAS

---

## 1. Training a Model

### Basic Training

```bash
predictmix train data.csv \
  --target-column disease_status \
  --model ensemble \
  --output-dir my_model
```

### Advanced Training with Feature Selection

```bash
predictmix train data.csv \
  --target-column disease_status \
  --model ensemble \
  --feature-selection lasso \
  --n-features 150 \
  --output-dir my_model \
  --plots
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Load YAML config file | None |
| `--model, -m` | Model type: `logreg`, `svm`, `rf`, `mlp`, `ensemble` | `ensemble` |
| `--feature-selection, -f` | Feature selection: `none`, `lasso`, `elasticnet`, `tree`, `chi2`, `pca` | `lasso` |
| `--n-features, -k` | Number of features to select | `100` |
| `--target-column, -y` | Target column name (binary: 0/1) | `y` |
| `--output-dir, -o` | Output directory | `predictmix_output` |
| `--export-predictions` | Export predictions CSV | Flag |
| `--plots/--no-plots` | Auto-generate ROC & PR plots | `--no-plots` |

### Output Files

```
my_model/
├── predictmix_model.joblib   # Trained model (reusable)
├── config.json               # Configuration used
├── metrics.json              # Performance metrics
└── predictions.csv           # Predictions with probabilities
```

#### metrics.json

```json
{
  "cv": {
    "accuracy": 0.78,
    "auc": 0.82,
    "precision_macro": 0.75,
    "recall_macro": 0.74,
    "f1_macro": 0.74
  },
  "test": {
    "accuracy": 0.80,
    "auc": 0.81,
    "precision_macro": 0.77,
    "recall_macro": 0.76,
    "f1_macro": 0.76
  }
}
```

---

## 2. Making Predictions

### Apply trained model to new data

```bash
predictmix predict my_model/predictmix_model.joblib new_patients.csv \
  --output predictions.csv
```

**Input:** CSV file with same features as training data (no target column needed)

**Output:** Original data + `risk_proba` column (predicted disease probability)

---

## 3. Generating Visualizations

### Generate All Plots

```bash
predictmix plot my_model/predictions.csv \
  --kind all \
  --output-dir my_plots
```

### Generate Specific Plot Types

```bash
# ROC and Precision-Recall curves
predictmix plot predictions.csv --kind rocpr --output-dir plots/

# Histograms
predictmix plot predictions.csv --kind hist --output-dir plots/

# Calibration curve
predictmix plot predictions.csv --kind calib --output-dir plots/

# Confusion matrix
predictmix plot predictions.csv --kind heatmap --output-dir plots/

# Scatter plot
predictmix plot predictions.csv --kind scatter --output-dir plots/
```

### Plot Options

| Option | Values | Description |
|--------|--------|-------------|
| `--kind, -k` | `all`, `rocpr`, `hist`, `scatter`, `heatmap`, `calib` | Plot type(s) to generate |
| `--output-dir, -o` | directory path | Output directory for plots |

---

## 4. Volcano Plots for GWAS

### Generate volcano plot from summary statistics

```bash
predictmix plot-volcano gwas_summary.csv \
  --effect-col beta \
  --pval-col pval \
  --output volcano.png
```

**Input format:**
```csv
snp_id,beta,pval,chr,position
rs123456,0.5,0.001,1,1000000
rs234567,-0.3,0.005,1,2000000
...
```

---

## Complete Example Workflow

```bash
# 1. Download test data
wget https://github.com/EtienneNtumba/predictmix/releases/download/v0.1.1/test_predictmix_data.csv

# 2. Train model with feature selection
predictmix train test_predictmix_data.csv \
  --target-column disease_status \
  --model ensemble \
  --feature-selection lasso \
  --n-features 8 \
  --output-dir results \
  --export-predictions

# 3. Generate all visualizations
predictmix plot results/predictions.csv \
  --kind all \
  --output-dir results/plots

# 4. Check performance metrics
cat results/metrics.json

# 5. Create new patient data for prediction (first 20 rows as example)
head -21 test_predictmix_data.csv > new_patients.csv

# 6. Make predictions on new patients
predictmix predict results/predictmix_model.joblib new_patients.csv \
  --output new_predictions.csv

# 7. View predictions
head new_predictions.csv
```

---

## Input Data Format

### Minimum Requirements

- One **binary target column** (0/1) - e.g., `disease_status`, `case_control`, `outcome`
- One or more **numeric feature columns** - e.g., PRS, age, BMI, lab values

### Example Dataset

```csv
disease_status,prs,age,sex,bmi,hemoglobin,family_history,smoking
0,0.12,35,0,22.5,14.2,0,0
1,1.45,29,1,27.1,12.1,1,1
0,-0.34,41,0,24.8,13.8,0,0
1,1.10,33,1,26.3,11.5,1,0
```

**If your target column has a different name:**

```bash
predictmix train data.csv --target-column case_control ...
```

---

## Model Comparison

Test all available models:

```bash
# Logistic Regression
predictmix train data.csv --model logreg --output-dir models/logreg

# Support Vector Machine
predictmix train data.csv --model svm --output-dir models/svm

# Random Forest
predictmix train data.csv --model rf --output-dir models/rf

# Neural Network (MLP)
predictmix train data.csv --model mlp --output-dir models/mlp

# Ensemble (Voting Classifier)
predictmix train data.csv --model ensemble --output-dir models/ensemble
```

Compare AUC scores:
```bash
for dir in models/*/; do
    echo "$dir:"
    grep '"auc"' "$dir/metrics.json" | head -1
done
```

---

## Feature Selection Comparison

```bash
# Test different feature selection methods
for method in none lasso elasticnet tree chi2 pca; do
    predictmix train data.csv \
        --feature-selection $method \
        --n-features 10 \
        --output-dir fs_$method
done

# Compare results
for dir in fs_*/; do
    echo "$dir:"
    grep '"test"' -A 5 "$dir/metrics.json"
done
```

---

## Performance Metrics

PredictMix reports the following metrics:

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve - discriminative ability |
| **Accuracy** | Overall classification accuracy |
| **Precision** | Positive Predictive Value |
| **Recall** | Sensitivity / True Positive Rate |
| **F1-Score** | Harmonic mean of precision and recall |

**All metrics reported for:**
- Cross-validation on training data
- Final test set

---

## Clinical Risk Stratification

Interpret predicted probabilities:

| Risk Score | Category | Clinical Action |
|------------|----------|-----------------|
| > 0.80 | Very High Risk | Immediate intervention |
| 0.60-0.80 | High Risk | Close monitoring |
| 0.40-0.60 | Moderate Risk | Regular follow-up |
| 0.20-0.40 | Low Risk | Standard screening |
| < 0.20 | Very Low Risk | Routine care |

```bash
# Analyze risk distribution
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('results/predictions.csv')
print(df.groupby('y_true')['risk_proba'].describe())
EOF
```

---

## Requirements

- **Python:** 3.8+
- **Dependencies** (auto-installed):
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - joblib
  - pyyaml
  - typer
  - matplotlib
  - lime
  - typing_extensions

---

## Project Structure

```
src/predictmix/
├── __init__.py
├── cli.py                # Command-line interface (Typer)
├── config.py             # Configuration management
├── data.py               # Data loading & preprocessing
├── feature_selection.py  # Feature selection methods
├── models.py             # Model factory (logreg, SVM, RF, MLP, ensemble)
├── pipeline.py           # Training & prediction pipeline
├── plots.py              # Visualization utilities
└── prs.py                # PRS utilities (extensible)
```

---

## Citation

If you use PredictMix in your research, please cite:

```bibtex
@software{predictmix2025,
  author = {Ntumba Kabongo, Etienne and Chimusa, Emile R.},
  title = {PredictMix: An Integrated Polygenic-Clinical Machine Learning Pipeline for Disease Risk Prediction},
  year = {2025},
  url = {https://github.com/EtienneNtumba/predictmix}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Support & Contact

- **Issues:** [GitHub Issues](https://github.com/EtienneNtumba/predictmix/issues)
- **Email:** [etienne.kabongo@mcgill.ca](mailto:etienne.kabongo@mcgill.ca)
- **Documentation:** [GitHub Wiki](https://github.com/EtienneNtumba/predictmix/wiki)

---

## Roadmap

### Planned Features

- [ ] SHAP explainability integration
- [ ] Multi-class classification support
- [ ] Deep learning models (Transformers)
- [ ] Integration with PRS-CS and LDpred
- [ ] Automated genotype QC and processing
- [ ] Nextflow/Snakemake workflows for HPC
- [ ] Interactive dashboards
- [ ] Model cards for transparency

---

## Acknowledgments

PredictMix was developed with support from:
- McGill University
- Northumbria University

Special thanks to all contributors and users providing feedback!

---

## Related Tools

- **PRSice-2:** PRS calculation - [GitHub](https://github.com/choishingwan/PRSice)
- **LDpred:** PRS adjustment - [GitHub](https://github.com/bvilhjal/ldpred)
- **PLINK:** Genetic data analysis - [Website](https://www.cog-genomics.org/plink/)

---

**Made with ❤️ for the genomics and precision medicine community**
