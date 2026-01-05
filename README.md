# BCHER2PET
Ternary Classification Prediction and Heterogeneity Quantification

# Project repo — integrated pipelines
This repository contains multiple analysis pipelines:
- python/habitat: voxel-level radiomics -> clustering -> habitat masks
- python/comparison: Friedman-Nemenyi model comparison utilities
- R/feature_selection: multiclass feature selection scripts
- R/ML_modeling: multiclass modeling pipeline

Usage (general)
- Install Python deps: pip install -r requirements.txt
- For R subprojects use renv: open R in subfolder and run renv::restore() or use the provided renv.lock
- See each subfolder for minimal README and usage examples.

Data / privacy
- Do NOT commit raw clinical data. Keep only synthetic / minimal example data in examples/.

License
- See LICENSE (MIT).
