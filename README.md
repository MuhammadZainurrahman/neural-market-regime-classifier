# Neural-Market-Regime-Classifier: Self-Supervised Market Intelligence

[![Research](https://img.shields.io/badge/Research-Self--Supervised-green.svg)](Abstract.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced framework for **Identifying Market Regimes** using **Self-Supervised Contrastive Learning**. This repository implements a deep learning pipeline that learns to group market states into latent clusters (Bull, Bear, Sideways, Volatile) without the need for manual labeling.

## 🔬 Core Methodology
- **Contrastive Learning**: Uses SimCLR-style temporal augmentations to learn robust market features.
- **Temporal 1D-Convolutions**: Extracts structural patterns from multi-channel OHLCV (Open, High, Low, Close, Volume) data.
- **Latent Manifold Analysis**: Projects temporal slices into a normalized manifold where distinct market regimes can be separated.

## 🛠 Project Structure
- `src/engine.py`: Core 1D-CNN Encoder and regime classification logic.
- `research/`: Experimental results and manifold visualizations (t-SNE/UMAP).
- `data/`: Data loading pipelines for CSV and WebSocket market feeds.

## 🚀 Quick Start
```bash
python src/engine.py
```

---

**Lead Researcher:** Muhammad Zainurrahman  
**Framework:** PyTorch | NumPy | Scikit-Learn
