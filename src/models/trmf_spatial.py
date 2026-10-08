"""
Temporal Regularized Matrix Factorization (TRMF) for Spatial Tourism Spillovers.
Based on Yu et al. (NeurIPS): Low-rank matrix decomposition Y(38 x T) ≈ W * X^T
capturing cross-border Schengen spillovers across 4 European tourism clusters without GNN.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TRMFSpatialSpillover:
    """
    TRMF model for cross-country spatial tourism forecasting.
    Decomposes the 38 European countries panel Y (N x T) into:
      - Spatial Factor Matrix W (N x rank): Latent country cluster embeddings
      - Temporal Latent Factor Matrix X (T x rank): Latent macro drivers across time
      - AR lag coefficients Theta: Modeling temporal dynamics of latent factors
    Forecasts multi-step spatial spillovers: Y_spatial = W * X_future^T
    """

    def __init__(
        self,
        rank: int = 4,              # 4 Natural European Tourism Clusters (Southern, Western, Northern, Eastern)
        ar_lags: Tuple[int, ...] = (1, 2, 12),  # Lag-1, Lag-2, and Seasonal Lag-12
        lambda_w: float = 1e-2,
        lambda_x: float = 1e-2,
        lambda_ar: float = 1e-1,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        self.rank = rank
        self.ar_lags = list(ar_lags)
        self.max_lag = max(ar_lags)
        self.lambda_w = lambda_w
        self.lambda_x = lambda_x
        self.lambda_ar = lambda_ar
        self.max_iter = max_iter
        self.tol = tol

        self.W: Optional[np.ndarray] = None  # (N, rank)
        self.X: Optional[np.ndarray] = None  # (T, rank)
        self.Theta: Optional[np.ndarray] = None  # (len(lags), rank)
        self.country_index_map: Dict[str, int] = {}
        self.mean_per_country: Optional[np.ndarray] = None
        self.std_per_country: Optional[np.ndarray] = None

    def _build_panel_matrix(
        self, df: pd.DataFrame, target_col: str = "arrivals_total", accom_type: str = "total"
    ) -> Tuple[np.ndarray, List[str], List[pd.Timestamp]]:
        """
        Pivots dataframe to an (N, T) matrix for the specified accommodation type.
        """
        if accom_type in df["accommodation_type"].values:
            sub = df[df["accommodation_type"] == accom_type].copy()
        else:
            sub = df.copy()
        pivot = sub.pivot(index="geo", columns="date", values=target_col)
        # Sort countries & dates
        countries = list(pivot.index)
        dates = list(pivot.columns)
        Y_raw = pivot.values.astype(np.float32)

        # Fill any missing values with country-wise mean
        col_means = np.nanmean(Y_raw, axis=1, keepdims=True)
        col_means = np.nan_to_num(col_means, nan=1.0)
        Y_raw = np.where(np.isnan(Y_raw), col_means, Y_raw)

        return Y_raw, countries, dates

    def fit(
        self, df: pd.DataFrame, target_col: str = "arrivals_total", accom_type: str = "total"
    ) -> "TRMFSpatialSpillover":
        """
        Fits TRMF via Alternating Least Squares (ALS) on the (N, T) country panel.
        Automatically adapts AR lags if temporal horizon T is short (e.g. annual panel).
        """
        Y_raw, countries, _ = self._build_panel_matrix(df, target_col, accom_type)
        self.country_index_map = {c: i for i, c in enumerate(countries)}
        N, T = Y_raw.shape

        # Adaptive AR lags for short panels (e.g. annual macroeconomic data)
        if T < 36 and max(self.ar_lags) >= 12:
            self.ar_lags = [l for l in self.ar_lags if l < max(2, T // 3)]
            if not self.ar_lags:
                self.ar_lags = [1, 2] if T >= 4 else [1]
            self.max_lag = max(self.ar_lags)

        # Standardize per country
        self.mean_per_country = np.mean(Y_raw, axis=1, keepdims=True)
        self.std_per_country = np.std(Y_raw, axis=1, keepdims=True) + 1e-5
        Y = (Y_raw - self.mean_per_country) / self.std_per_country

        # Initialize W and X using truncated SVD
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        self.W = U[:, :self.rank] * np.sqrt(s[:self.rank])
        self.X = Vt[:self.rank, :].T * np.sqrt(s[:self.rank])  # (T, rank)
        self.Theta = np.zeros((len(self.ar_lags), self.rank), dtype=np.float32)

        # Alternating Least Squares optimization
        prev_loss = float("inf")
        for iteration in range(self.max_iter):
            # 1. Update W: min ||Y - W X^T||_F^2 + lambda_w ||W||_F^2
            # W = Y X (X^T X + lambda_w I)^(-1)
            reg_x = self.X.T @ self.X + self.lambda_w * np.eye(self.rank)
            self.W = Y @ self.X @ np.linalg.inv(reg_x)

            # 2. Update AR coefficients Theta on temporal latent factors X
            # For each factor r in [0, rank):
            for r in range(self.rank):
                x_r = self.X[:, r]
                # Build lag matrix
                lag_mat = []
                target_vec = []
                for t in range(self.max_lag, T):
                    lag_mat.append([x_r[t - l] for l in self.ar_lags])
                    target_vec.append(x_r[t])
                if len(lag_mat) > 0:
                    L_mat = np.array(lag_mat)
                    y_vec = np.array(target_vec)
                    # Ridge regression with strong regularization for stationary AR parameters
                    theta_r = np.linalg.solve(
                        L_mat.T @ L_mat + 2.0 * np.eye(len(self.ar_lags)), L_mat.T @ y_vec
                    )
                    # Clip theta to ensure stability
                    theta_r = np.clip(theta_r, -0.6, 0.6)
                    self.Theta[:, r] = theta_r

            # 3. Update X with temporal regularization
            grad_x = -2 * (Y - self.W @ self.X.T).T @ self.W + 2 * self.lambda_x * self.X
            lr = 0.05 / (iteration + 1) ** 0.5
            self.X = np.clip(self.X - lr * grad_x, -4.0, 4.0)

            # Check convergence
            reconstruction = self.W @ self.X.T
            loss = np.mean((Y - reconstruction) ** 2)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def forecast(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """
        Rolls forward the latent temporal factors X using the estimated AR dynamics,
        then projects into country space: Y_future = W * X_future^T.
        Returns: {country_code: array of length horizon}
        """
        if self.W is None or self.X is None or self.Theta is None:
            raise ValueError("Model must be fitted before forecasting.")

        T, rank = self.X.shape
        X_extended = list(self.X)

        # Roll forward H steps with standard exponential damping for stationarity
        damping = 0.96
        for h in range(horizon):
            next_factors = np.zeros(rank, dtype=np.float32)
            t_curr = T + h
            for r in range(rank):
                val = 0.0
                for lag_idx, l in enumerate(self.ar_lags):
                    prev_idx = t_curr - l
                    if prev_idx < len(X_extended):
                        val += self.Theta[lag_idx, r] * X_extended[prev_idx][r]
                # Damping factor to prevent explosive multi-step drift
                next_factors[r] = np.clip(val * (damping ** (h + 1)), -3.0, 3.0)
            X_extended.append(next_factors)

        X_future = np.array(X_extended[T:], dtype=np.float32)  # (H, rank)
        Y_future_norm = self.W @ X_future.T                     # (N, H)
        # Denormalize
        Y_future = Y_future_norm * self.std_per_country + self.mean_per_country
        # Physical bounds: non-negative and capped at 3.0x historical country mean
        max_bound = np.maximum(self.mean_per_country * 3.0, 1000.0)
        Y_future = np.clip(Y_future, a_min=0, a_max=max_bound)

        results = {}
        for country, idx in self.country_index_map.items():
            results[country] = Y_future[idx, :]
        return results
