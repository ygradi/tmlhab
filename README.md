# Introduction
Ternary Classification Prediction and Heterogeneity Quantification for HER2 in Breast Cancer using HER2-targeted PET/CT imaging

This repository contains multiple analysis pipelines:
- python/habitat: voxel-level radiomics -> clustering -> habitat masks
- python/comparison: Friedman-Nemenyi model comparison utilities
- R/feature_selection: multiclass feature selection scripts
- R/R/ternary_modeling_fig_shap.R: multiclass modeling and interpretation pipeline

Usage
- Install Python deps: pip install -r requirements.txt
- For R subprojects use renv: open R in subfolder and run renv::restore() or use the provided renv.lock
