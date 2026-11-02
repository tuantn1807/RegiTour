"""Residual-based split conformal intervals for point forecasts."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def split_validation_temporally(
    validation_df: pd.DataFrame,
    calibration_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation by date into tuning and later calibration periods.

    The two returned frames never share a timestamp. With fewer than two
    timestamps, all validation rows are reserved for calibration and the
    tuning frame is empty.
    """
    if validation_df.empty:
        empty = validation_df.iloc[0:0].copy()
        return empty, validation_df.copy()

    if "date" not in validation_df.columns:
        raise ValueError("Temporal validation split requires a 'date' column.")

    dates = pd.Index(pd.to_datetime(validation_df["date"]).dropna().unique()).sort_values()
    if len(dates) < 2:
        return validation_df.iloc[0:0].copy(), validation_df.copy()

    fraction = float(np.clip(calibration_fraction, 0.0, 1.0))
    n_calibration_dates = max(1, int(np.ceil(len(dates) * fraction)))
    n_calibration_dates = min(n_calibration_dates, len(dates) - 1)
    split_date = dates[len(dates) - n_calibration_dates]
    row_dates = pd.to_datetime(validation_df["date"])

    tuning_df = validation_df.loc[row_dates < split_date].copy()
    calibration_df = validation_df.loc[row_dates >= split_date].copy()
    return tuning_df, calibration_df


class ResidualConformalForecaster:
    """Calibrate intervals from out-of-sample residuals of the point forecaster.

    This class does not train a second prediction model. For each target it
    stores the finite-sample conformal quantile of absolute validation errors,
    then applies that margin to the corresponding point forecasts.
    """

    def __init__(self, target_coverage: float = 0.90):
        if not 0.0 < target_coverage < 1.0:
            raise ValueError("target_coverage must be between 0 and 1.")
        self.target_coverage = float(target_coverage)
        self.conformal_q_arr = float("inf")
        self.conformal_q_nig = float("inf")
        self.calibration_size_arr = 0
        self.calibration_size_nig = 0
        self.is_calibrated = False

    @staticmethod
    def _finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        n = len(scores)
        if n == 0:
            return float("inf")

        # Split-conformal order statistic: ceil((n + 1) * coverage).
        rank = int(np.ceil((n + 1) * coverage))
        if rank > n:
            return float("inf")
        return float(np.partition(scores, rank - 1)[rank - 1])

    def calibrate_residuals(
        self,
        y_true_arr: np.ndarray,
        y_pred_arr: np.ndarray,
        y_true_nig: np.ndarray,
        y_pred_nig: np.ndarray,
    ) -> "ResidualConformalForecaster":
        """Calibrate from paired held-out truths and main-model point forecasts."""
        pairs = (
            ("arrivals", y_true_arr, y_pred_arr),
            ("nights", y_true_nig, y_pred_nig),
        )
        quantiles = {}
        sizes = {}
        for name, y_true, y_pred in pairs:
            truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
            prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            if len(truth) != len(prediction):
                raise ValueError(f"{name}: truth and prediction lengths do not match.")
            valid = np.isfinite(truth) & np.isfinite(prediction)
            scores = np.abs(truth[valid] - prediction[valid])
            quantiles[name] = self._finite_sample_quantile(scores, self.target_coverage)
            sizes[name] = int(len(scores))

        self.conformal_q_arr = quantiles["arrivals"]
        self.conformal_q_nig = quantiles["nights"]
        self.calibration_size_arr = sizes["arrivals"]
        self.calibration_size_nig = sizes["nights"]
        self.is_calibrated = self.calibration_size_arr > 0 and self.calibration_size_nig > 0
        return self

    @staticmethod
    def _intervals(point: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        lower = np.maximum(0.0, point - margin)
        upper = np.maximum(0.0, point + margin)
        return lower.astype(np.float32), point.astype(np.float32), upper.astype(np.float32)

    def predict_conformal_intervals(
        self,
        y_pred_arr: np.ndarray,
        y_pred_nig: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return nonnegative conformal lower/point/upper forecast limits."""
        return {
            "arrivals": self._intervals(y_pred_arr, self.conformal_q_arr),
            "nights": self._intervals(y_pred_nig, self.conformal_q_nig),
        }


# Legacy import alias; this class now exposes the residual-calibration API.
DeepConformalQuantileForecaster = ResidualConformalForecaster
