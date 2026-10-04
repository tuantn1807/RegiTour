"""
Dynamic Mixture-of-Experts (MoE) Gating Network.
Arbiter combining the 3 primary prediction engines:
  1. Y_trend (CatBoost-Regime Macro Engine)
  2. Y_seasonal (M-RA-BiLSTM Calendar & Shock Engine)
  3. Y_spatial (TRMF Schengen Spatial Spillover Engine)
Computes adaptive weights: w1(t) + w2(t) + w3(t) = 1 via context-conditioned Softmax.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DynamicMoEGatingNetwork:
    """
    Learns dynamic gating weights w1, w2, w3 conditioned on:
      - Regime probabilities [P_normal, P_disruption, P_recovery]
      - Calendar seasonality indicators (is_peak_summer, is_winter_holiday, holiday_days)
      - Month indices
    """

    def __init__(
        self,
        input_dim: int = 7,  # [p_normal, p_disruption, p_recovery, is_summer, is_winter, holiday_days, month_sin]
        hidden_dim: int = 32,
        num_experts: int = 3,
        learning_rate: float = 1e-3,
        epochs: int = 100,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.epochs = epochs

        if HAS_TORCH:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
            )
            self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        else:
            self.mlp = None

    def fit(
        self,
        context_features: np.ndarray,      # (N, input_dim)
        expert_preds: np.ndarray,          # (N, num_experts)
        y_true: np.ndarray,                # (N,)
    ) -> "DynamicMoEGatingNetwork":
        """
        Trains gating network by minimizing Huber loss between ensemble prediction and ground truth.
        """
        if not HAS_TORCH:
            return self

        self.mlp.train()
        X_t = torch.tensor(context_features, dtype=torch.float32)
        E_t = torch.tensor(expert_preds, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32).unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(X_t, E_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        huber = nn.SmoothL1Loss()

        for _ in range(self.epochs):
            for bx, be, by in loader:
                self.optimizer.zero_grad()
                logits = self.mlp(bx)
                weights = F.softmax(logits, dim=-1)  # (B, 3)
                ensemble_pred = torch.sum(weights * be, dim=-1, keepdim=True)
                loss = huber(ensemble_pred, by)
                loss.backward()
                self.optimizer.step()

        return self

    def predict_weights(self, context_features: np.ndarray) -> np.ndarray:
        """
        Returns dynamic weights matrix: (N, 3) where sum(axis=1) == 1.0.
        """
        if not HAS_TORCH:
            # Heuristic default weighting:
            # High summer/holiday -> more seasonal; Disruption -> more macro trend
            weights = np.ones((len(context_features), self.num_experts), dtype=np.float32)
            weights[:, 0] = 0.35  # CatBoost trend
            weights[:, 1] = 0.45  # M-RA-BiLSTM seasonal
            weights[:, 2] = 0.20  # TRMF spatial
            return weights

        self.mlp.eval()
        with torch.no_grad():
            X_t = torch.tensor(context_features, dtype=torch.float32)
            logits = self.mlp(X_t)
            weights = F.softmax(logits, dim=-1).numpy()
            return weights

    def combine_experts(
        self,
        context_features: np.ndarray,
        y_trend: np.ndarray,
        y_seasonal: np.ndarray,
        y_spatial: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensembles forecasts using dynamic gating weights:
        Y_point = w1 * Y_trend + w2 * Y_seasonal + w3 * Y_spatial.
        Returns: (Y_point, weights)
        """
        weights = self.predict_weights(context_features)
        y_point = (
            weights[:, 0] * y_trend +
            weights[:, 1] * y_seasonal +
            weights[:, 2] * y_spatial
        )
        return np.clip(y_point, a_min=0, a_max=None), weights
