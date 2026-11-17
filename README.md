# Multi-Class Trip Label Prediction (Ride-Hailing Analytics)

![ci](https://github.com/henryferdianto/trip-label-classification-datavers/actions/workflows/ci.yml/badge.svg)

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


<!-- completed -->

## Quickstart with Sample Data

The repository ships a synthetic data generator so the full pipeline can be run
without any private dataset:

```bash
make sample && python -m src.train
python -m src.run_demo
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Project Status

This repository contains the project documentation and a runnable reference
implementation. A synthetic data generator is included so every pipeline stage
can be executed end-to-end; replace the sample data with your own datasets in
`data/raw/` to reproduce real results.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
