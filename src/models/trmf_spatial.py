"""
Temporal Regularized Matrix Factorization (TRMF) for Spatial Tourism Spillovers.
Based on Yu et al. (NeurIPS): Low-rank matrix decomposition Y(38 x T) ≈ W * X^T
capturing cross-border Schengen spillovers across 4 European tourism clusters without GNN.

Supports dual matrix factorization:
  1. Y_arr (Arrivals) ≈ W_arr * X_arr^T : Captures visitor flow across borders.
  2. Y_nig (Overnight Stays) ≈ W_nig * X_nig^T : Independently learns spatial spillover
     and duration patterns (Length of Stay / LoS dynamics) across destinations.
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

        # Primary target matrices (Arrivals / Y_arr)
        self.W: Optional[np.ndarray] = None  # (N, rank)
        self.X: Optional[np.ndarray] = None  # (T, rank)
        self.Theta: Optional[np.ndarray] = None  # (len(lags), rank)
        self.country_index_map: Dict[str, int] = {}
        self.mean_per_country: Optional[np.ndarray] = None
        self.std_per_country: Optional[np.ndarray] = None

        # Secondary target matrices (Overnight Stays / Y_nig) for dual spatial modeling
        self.W_nig: Optional[np.ndarray] = None  # (N, rank)
        self.X_nig: Optional[np.ndarray] = None  # (T, rank)
        self.Theta_nig: Optional[np.ndarray] = None  # (len(lags), rank)
        self.mean_per_country_nig: Optional[np.ndarray] = None
        self.std_per_country_nig: Optional[np.ndarray] = None
        self.is_dual: bool = False

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

    def _optimize_factors(
        self, Y_raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs Alternating Least Squares (ALS) optimization on a standardized panel matrix.
        Returns: (W, X, Theta, mean_per_country, std_per_country)
        """
        N, T = Y_raw.shape

        # Adaptive AR lags for short panels (e.g. annual macroeconomic data)
        ar_lags = list(self.ar_lags)
        if T < 36 and max(ar_lags) >= 12:
            ar_lags = [l for l in ar_lags if l < max(2, T // 3)]
            if not ar_lags:
                ar_lags = [1, 2] if T >= 4 else [1]
        max_lag = max(ar_lags)

        # Standardize per country
        mean_per_country = np.mean(Y_raw, axis=1, keepdims=True)
        std_per_country = np.std(Y_raw, axis=1, keepdims=True) + 1e-5
        Y = (Y_raw - mean_per_country) / std_per_country

        # Initialize W and X using truncated SVD
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        W = U[:, :self.rank] * np.sqrt(s[:self.rank])
        X = Vt[:self.rank, :].T * np.sqrt(s[:self.rank])  # (T, rank)
        Theta = np.zeros((len(ar_lags), self.rank), dtype=np.float32)

        prev_loss = float("inf")
        for iteration in range(self.max_iter):
            # 1. Update W: min ||Y - W X^T||_F^2 + lambda_w ||W||_F^2
            reg_x = X.T @ X + self.lambda_w * np.eye(self.rank)
            W = Y @ X @ np.linalg.inv(reg_x)

            # 2. Update AR coefficients Theta on temporal latent factors X
            for r in range(self.rank):
                x_r = X[:, r]
                lag_mat = []
                target_vec = []
                for t in range(max_lag, T):
                    lag_mat.append([x_r[t - l] for l in ar_lags])
                    target_vec.append(x_r[t])
                if len(lag_mat) > 0:
                    L_mat = np.array(lag_mat)
                    y_vec = np.array(target_vec)
                    theta_r = np.linalg.solve(
                        L_mat.T @ L_mat + 2.0 * np.eye(len(ar_lags)), L_mat.T @ y_vec
                    )
                    theta_r = np.clip(theta_r, -0.6, 0.6)
                    Theta[:, r] = theta_r

            # 3. Update X with temporal regularization
            grad_x = -2 * (Y - W @ X.T).T @ W + 2 * self.lambda_x * X
            lr = 0.05 / (iteration + 1) ** 0.5
            X = np.clip(X - lr * grad_x, -4.0, 4.0)

            # Check convergence
            reconstruction = W @ X.T
            loss = np.mean((Y - reconstruction) ** 2)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return W, X, Theta, mean_per_country, std_per_country

    def _forecast_factors(
        self,
        W: np.ndarray,
        X: np.ndarray,
        Theta: np.ndarray,
        mean_per_country: np.ndarray,
        std_per_country: np.ndarray,
        horizon: int = 12,
    ) -> Dict[str, np.ndarray]:
        """
        Projects latent factors forward via estimated AR dynamics and reconstructs country forecasts.
        """
        T, rank = X.shape
        X_extended = list(X)

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
                        val += Theta[lag_idx, r] * X_extended[prev_idx][r]
                next_factors[r] = np.clip(val * (damping ** (h + 1)), -3.0, 3.0)
            X_extended.append(next_factors)

        X_future = np.array(X_extended[T:], dtype=np.float32)  # (H, rank)
        Y_future_norm = W @ X_future.T                         # (N, H)
        # Denormalize
        Y_future = Y_future_norm * std_per_country + mean_per_country
        max_bound = np.maximum(mean_per_country * 3.0, 1000.0)
        Y_future = np.clip(Y_future, a_min=0, a_max=max_bound)

        results = {}
        for country, idx in self.country_index_map.items():
            results[country] = Y_future[idx, :]
        return results

    def fit(
        self, df: pd.DataFrame, target_col: str = "arrivals_total", accom_type: str = "total"
    ) -> "TRMFSpatialSpillover":
        """
        Fits TRMF via Alternating Least Squares (ALS) for a single target series (e.g. arrivals).
        """
        Y_raw, countries, _ = self._build_panel_matrix(df, target_col, accom_type)
        self.country_index_map = {c: i for i, c in enumerate(countries)}
        self.W, self.X, self.Theta, self.mean_per_country, self.std_per_country = (
            self._optimize_factors(Y_raw)
        )
        self.is_dual = False
        return self

    def fit_dual(
        self,
        df: pd.DataFrame,
        target_arr: str = "arrivals_total",
        target_nig: str = "nights_total",
        accom_type: str = "total",
    ) -> "TRMFSpatialSpillover":
        """
        Fits dual TRMF factorizations independently for both:
          1. Arrivals (Y_arr): captures cross-border visitor flows.
          2. Overnight Stays (Y_nig): captures destination length of stay (LoS) duration
             and independent spatial spillovers without assuming a linear scalar ratio.
        """
        Y_raw_arr, countries, _ = self._build_panel_matrix(df, target_arr, accom_type)
        Y_raw_nig, _, _ = self._build_panel_matrix(df, target_nig, accom_type)
        self.country_index_map = {c: i for i, c in enumerate(countries)}

        # Fit independent spatial factor decomposition for Arrivals
        self.W, self.X, self.Theta, self.mean_per_country, self.std_per_country = (
            self._optimize_factors(Y_raw_arr)
        )

        # Fit independent spatial factor decomposition for Overnight Stays (Ynig)
        self.W_nig, self.X_nig, self.Theta_nig, self.mean_per_country_nig, self.std_per_country_nig = (
            self._optimize_factors(Y_raw_nig)
        )

        self.is_dual = True
        return self

    def forecast(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """
        Forecasts spatial spillovers for the primary target (Arrivals).
        Returns: {country_code: array of length horizon}
        """
        if self.W is None or self.X is None or self.Theta is None:
            raise ValueError("Model must be fitted before forecasting.")
        return self._forecast_factors(
            self.W, self.X, self.Theta, self.mean_per_country, self.std_per_country, horizon
        )

    def forecast_dual(
        self, horizon: int = 12
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Forecasts multi-horizon spatial spillovers independently for both:
          - Arrivals: using W_arr, X_arr, Theta_arr
          - Overnight Stays: using W_nig, X_nig, Theta_nig
        Returns: (forecasts_arrivals, forecasts_nights)
        """
        if not self.is_dual or self.W_nig is None or self.X_nig is None:
            # Fallback to single target if fit_dual was not called
            res_arr = self.forecast(horizon)
            res_nig = {c: arr.copy() * 2.5 for c, arr in res_arr.items()}
            return res_arr, res_nig

        res_arr = self._forecast_factors(
            self.W, self.X, self.Theta, self.mean_per_country, self.std_per_country, horizon
        )
        res_nig = self._forecast_factors(
            self.W_nig, self.X_nig, self.Theta_nig, self.mean_per_country_nig, self.std_per_country_nig, horizon
        )
        return res_arr, res_nig
