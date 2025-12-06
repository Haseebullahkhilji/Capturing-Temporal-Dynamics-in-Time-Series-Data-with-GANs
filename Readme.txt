# TimeGAN Experiment (ready-to-run)

## Overview
This repo contains scripts to preprocess time series data, train a TimeGAN-style model, generate synthetic sequences, and evaluate them with multiple metrics (MMD, DTW, ACF/PACF, predictive utility).

## Files
- `preprocess.py` — Preprocess CSV to windowed sequences.
- `train_timegan.py` — Train TimeGAN-like model (GRU-based).
- `generate.py` — Generate synthetic sequences (unscaled).
- `evaluate.py` — Evaluate synthetic vs real (MMD, DTW, ACF/PACF, predictive).
- `requirements.txt` — Python packages.

## Example pipeline
1. Put your raw CSV (must include `Date`, and price columns `Open,High,Low,Close,Volume`) in `data/raw/`.
2. Preprocess:
3. Train:
4. Generate:
5. Evaluate:
