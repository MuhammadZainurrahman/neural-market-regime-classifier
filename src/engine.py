import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class RegimeEncoder(nn.Module):
    """
    1D-Convolutional Encoder for temporal feature extraction from market data.
    Designed for Self-Supervised contrastive learning of market regimes.
    """
    def __init__(self, input_channels: int, latent_dim: int = 32):
        super(RegimeEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        latent = self.fc(x)
        return F.normalize(latent, dim=1) # Normalize for contrastive loss

class MarketRegimeClassifier:
    """
    Downstream classifier that maps latent representations to specific regimes.
    Regimes: [0: Bull, 1: Bear, 2: Sideways, 3: Volatile]
    """
    def __init__(self, latent_dim: int = 32, num_regimes: int = 4):
        self.encoder = RegimeEncoder(input_channels=5, latent_dim=latent_dim)
        self.head = nn.Linear(latent_dim, num_regimes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.encoder(x)
            logits = self.head(latent)
            return F.softmax(logits, dim=1)

def generate_synthetic_market_data(samples: int = 100, length: int = 60):
    # Generates synthetic (batch, channels, length) data
    # Channels: [Open, High, Low, Close, Volume]
    return torch.randn(samples, 5, length)

if __name__ == "__main__":
    classifier = MarketRegimeClassifier()
    data = generate_synthetic_market_data(samples=5)
    
    predictions = classifier.predict(data)
    print("Market Regime Probabilities (Sample 0):")
    regimes = ["Bull", "Bear", "Sideways", "Volatile"]
    for i, prob in enumerate(predictions[0]):
        print(f"  {regimes[i]}: {prob.item():.4f}")
    
    print("\nRegimeEncoder Initialized. Ready for Contrastive Pre-training.")
