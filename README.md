# Multi-Class Trip Label Prediction (Ride-Hailing Analytics)

> **DataVers ANAVA #20 — spatiotemporal trip classification**

## Overview

A supervised multi-class classification pipeline predicting trip labels for ride-hailing services using spatiotemporal, route and behavioral features, with feature selection, imbalance handling and hyperparameter tuning.

## Approach

1. **Geospatial & temporal feature engineering**.
2. **Class-imbalance handling** and feature selection.
3. **Model benchmark** with gradient boosting and calibration.
4. **Evaluation** focused on macro F1 against baselines.

## Repository Structure

```
trip-label-classification-datavers/
├── src/features.py
├── src/train.py
├── src/evaluate.py
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `Optuna`

## Getting Started

```bash
git clone https://github.com/henryferdianto/trip-label-classification-datavers.git
cd trip-label-classification-datavers
pip install -r requirements.txt
```

```bash
python src/train.py --input data/raw/trips.csv --target trip_label
python src/evaluate.py --pred reports/pred.csv
```

## Project Status

This repository contains the project documentation and a runnable implementation
skeleton. Scripts are structured so that each pipeline stage can be executed and
extended independently. Update the configuration and data paths to run against
your own datasets.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
