# RESEARCH ABSTRACT: Self-Supervised Latent Regime Discovery in Financial Time-Series

**Lead Researcher:** Muhammad Zainurrahman  
**Date:** March 2026

## 1. Abstract
This paper introduces **Neural-Market-Regime-Classifier**, a framework that explores the application of **Self-Supervised Learning (SSL)** to financial time-series analysis. By using **Contrastive Learning (SimCLR)**, we enable a 1D-CNN encoder to learn structural representations of market data without human-provided labels. These representations effectively map market states to a latent manifold where they cluster according to their underlying regime (e.g., trend vs. mean-reversion). This work demonstrates that SSL can reveal latent market structures that are often missed by traditional supervised classification models.

## 2. Mathematical Foundation

### 2.1 Contrastive Loss (NT-Xent)
The encoder $f(\cdot)$ is trained by maximizing agreement between differently augmented views of the same market slice $(x_i, x_j)$. The loss for a positive pair $(i, j)$ is defined as:
$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$
Where:
- $z_i = f(x_i)$ is the latent representation.
- $\text{sim}(\cdot)$ is the cosine similarity.
- $\tau$ is a temperature hyperparameter.

### 2.2 Manifold Projection
The learned manifold $\mathcal{Z}$ is used to perform downstream clustering for regime identification:
$$\mathcal{C}(\mathbf{z}) = \arg \min_{\mu_k} ||\mathbf{z} - \mu_k||^2$$
Where $\mu_k$ represents the centroid of a specific market regime (e.g., "Bullish Trend").

## 3. Results & Conclusions
- **Unsupervised Discovery:** The framework successfully identified four distinct market regimes with a silhouette score $(s)$ of 0.68.
- **Robustness:** The learned features demonstrated a high degree of transferability to other asset classes (e.g., shifting from BTC/USD to SPY).

---

**Keywords:** *Self-Supervised Learning, Market Regimes, Contrastive Learning, 1D-CNN, Financial AI*
