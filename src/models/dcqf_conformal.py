"""
Deep Conformal Quantile Forecaster (DCQF).
Engine 4 in the RA-HE-BiLSTM Architecture.
Specializes in Uncertainty Quantification and Risk Interval Prediction [P10, P50, P90].
Combines:
  1. Multi-quantile asymmetric Pinball Loss (Quantile Regression)
  2. Conformal Prediction Calibration with distribution-free 90% coverage guarantee.
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


def pinball_loss(y_pred: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    """
    Computes Pinball Loss for a given quantile alpha.
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


class QuantileMLP:
    """
    Multi-output Quantile MLP predicting quantiles [P10, P50, P90] simultaneously.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        learning_rate: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 64,
    ):
        self.input_dim = input_dim
        self.quantiles = list(quantiles)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if HAS_TORCH:
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_dim = h
            # Output 3 quantiles for each target
            layers.append(nn.Linear(prev_dim, len(quantiles)))
            self.network = nn.Sequential(*layers)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            self.network = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not HAS_TORCH:
            return self

        self.network.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.network(batch_x)  # (B, 3)
                # Compute Pinball loss across quantiles
                total_loss = 0.0
                for q_idx, q in enumerate(self.quantiles):
                    pred_q = preds[:, q_idx:q_idx+1]
                    diff = batch_y - pred_q
                    loss_q = torch.mean(torch.max(q * diff, (q - 1.0) * diff))
                    total_loss = total_loss + loss_q

                total_loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not HAS_TORCH:
            # Fallback simple quantile estimation
            median = np.median(X, axis=1, keepdims=True)
            return np.hstack([median * 0.85, median, median * 1.15])

        self.network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.network(X_t).numpy()
            # Enforce monotonic quantiles: P10 <= P50 <= P90
            preds[:, 0] = np.minimum(preds[:, 0], preds[:, 1])
            preds[:, 2] = np.maximum(preds[:, 2], preds[:, 1])
            preds = np.clip(preds, a_min=0, a_max=None)
            return preds


class DeepConformalQuantileForecaster:
    """
    DCQF Engine: Trains quantile neural forecaster and performs conformal calibration
    to guarantee finite-sample distribution-free coverage.
    """

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        target_coverage: float = 0.90,
    ):
        self.quantiles = quantiles
        self.target_coverage = target_coverage
        self.model_arr: Optional[QuantileMLP] = None
        self.model_nig: Optional[QuantileMLP] = None
        self.conformal_q_arr: float = 0.0
        self.conformal_q_nig: float = 0.0

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train_arr: np.ndarray,
        y_train_nig: np.ndarray,
        X_calib: np.ndarray,
        y_calib_arr: np.ndarray,
        y_calib_nig: np.ndarray,
    ) -> "DeepConformalQuantileForecaster":
        """
        1. Trains QuantileMLP on training set.
        2. Computes conformity non-conformity scores on calibration set:
           s_i = max(y_i - q_90, q_10 - y_i)
        3. Sets conformal margin q_hat at ceil((N+1)*(1-alpha))/N.
        """
        feat_dim = X_train.shape[1]
        self.model_arr = QuantileMLP(input_dim=feat_dim, quantiles=self.quantiles)
        self.model_nig = QuantileMLP(input_dim=feat_dim, quantiles=self.quantiles)

        self.model_arr.fit(X_train, y_train_arr)
        self.model_nig.fit(X_train, y_train_nig)

        # Conformal calibration step
        if len(X_calib) > 0:
            preds_calib_arr = self.model_arr.predict(X_calib)
            p10_arr, p90_arr = preds_calib_arr[:, 0], preds_calib_arr[:, 2]
            scores_arr = np.maximum(y_calib_arr - p90_arr, p10_arr - y_calib_arr)
            n_calib = len(scores_arr)
            alpha_error = 1.0 - self.target_coverage
            q_idx = int(np.ceil((n_calib + 1) * (1.0 - alpha_error))) - 1
            q_idx = max(0, min(q_idx, n_calib - 1))
            self.conformal_q_arr = float(np.sort(scores_arr)[q_idx])

            preds_calib_nig = self.model_nig.predict(X_calib)
            p10_nig, p90_nig = preds_calib_nig[:, 0], preds_calib_nig[:, 2]
            scores_nig = np.maximum(y_calib_nig - p90_nig, p10_nig - y_calib_nig)
            self.conformal_q_nig = float(np.sort(scores_nig)[q_idx])

        return self

    def predict_conformal_intervals(
        self, X: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns guaranteed intervals:
          - arrivals: (P10_conformal, P50, P90_conformal)
          - nights: (P10_conformal, P50, P90_conformal)
        """
        preds_arr = self.model_arr.predict(X)
        preds_nig = self.model_nig.predict(X)

        # Apply conformal adjustments
        p10_arr_conf = np.clip(preds_arr[:, 0] - self.conformal_q_arr, a_min=0, a_max=None)
        p50_arr = preds_arr[:, 1]
        p90_arr_conf = np.clip(preds_arr[:, 2] + self.conformal_q_arr, a_min=0, a_max=None)

        p10_nig_conf = np.clip(preds_nig[:, 0] - self.conformal_q_nig, a_min=0, a_max=None)
        p50_nig = preds_nig[:, 1]
        p90_nig_conf = np.clip(preds_nig[:, 2] + self.conformal_q_nig, a_min=0, a_max=None)

        return {
            "arrivals": (p10_arr_conf, p50_arr, p90_arr_conf),
            "nights": (p10_nig_conf, p50_nig, p90_nig_conf),
        }
