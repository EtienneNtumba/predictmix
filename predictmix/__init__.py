"""
PredictMix – Integrated Polygenic + Clinical Disease Risk Prediction Pipeline
=============================================================================
Version : 0.2.0
Authors : Etienne Ntumba Kabongo (McGill University)
          Prof. Dr Emile R. Chimusa (Northumbria University)

New in 0.2.0
------------
Models (18 total):
  Classical  : logreg, bayesian, svm, rf, mlp, adaboost, bagging, ensemble
  Survival   : coxph, deepsurv
  Stratif.   : gmm, kmeans_risk
  Transfer   : transfer
  Deep (NumPy): rnn, lstm, gru, cnn1d, transformer, sedl

Feature selection (15 total):
  Filter     : none, pearson, chi2, infogain
  Embedded   : lasso, ridge, elasticnet, tree
  Reduction  : pca
  Wrapper    : rfe
  Deep DL    : dnn_l1, gated, autoencoder, stability, stacked_dl

PRS:
  - Standard weighted sum
  - H-PRS Fusion: LDpred2 + Lassosum2 + PRS-CSx

CLI:
  - train, predict, plot, benchmark, plot-volcano
"""
from __future__ import annotations
from .pipeline import PredictMixPipeline

__all__ = ["PredictMixPipeline"]
__version__ = "0.2.0"
__author__ = "Etienne Ntumba Kabongo"
__credits__ = [
    "Etienne Ntumba Kabongo (McGill University)",
    "Prof. Dr Emile R. Chimusa (Northumbria University)",
]
