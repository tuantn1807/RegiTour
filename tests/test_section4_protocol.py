"""Scientific invariants for matched forecast origins and scoring eligibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.section4_experiments import (
    build_core_inputs, get_scores, prepare_directory, project,
)
from scripts.reviewer_experiments import choose_weight
from src.models.ra_exo_dlinear import RAExoDLinearForecaster


class DummyForecaster:
    """Expose the input-building interface without fitting a neural network."""

    lookback_len = 24
    forecast_horizon = 12
    device = torch.device("cpu")
    future_exo_cols = ["month"]
    past_exo_cols = ["macro"]
    future_exo_mean = np.zeros(1)
    future_exo_scale = np.ones(1)
    past_exo_mean = np.zeros(1)
    past_exo_scale = np.ones(1)

    def _baseline_sequence(self, geo: str, accom: str, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(12), np.ones(12)

    def _feature_matrix(self, rows: pd.DataFrame, cols: list[str], mean: np.ndarray,
                        scale: np.ndarray) -> np.ndarray:
        return rows[cols].to_numpy(dtype=np.float32)


def make_frame() -> pd.DataFrame:
    """Create a regular panel with one full forecast year after two history years."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame({"date": dates, "geo": "AA", "accommodation_type": "total",
                         "arrivals_total": np.arange(36) + 1.0, "nights_total": np.arange(36) + 2.0,
                         "month": dates.month, "macro": np.arange(36), "regime_state": 0})


def test_future_outcomes_cannot_change_forecast_inputs() -> None:
    """Future actual targets and historical-only covariates must not enter an origin."""
    frame = make_frame()
    rows = frame.iloc[24:].copy().reset_index(drop=True)
    rows["origin"] = pd.Timestamp("2021-12-31 23:59:59")
    original, _ = build_core_inputs(DummyForecaster(), frame, rows)
    changed = frame.copy()
    changed.loc[changed.date.ge("2022-01-01"), ["arrivals_total", "nights_total", "macro"]] = 1e15
    alternative, _ = build_core_inputs(DummyForecaster(), changed, rows)
    for a, b in zip(original, alternative):
        assert torch.equal(a, b)
    assert original[0][0, -1, 0] == 24


def test_zero_targets_are_scored_but_unobserved_rows_are_not() -> None:
    """The original pair mask controls scoring independently of filled targets."""
    rows = pd.DataFrame({"observed_pair": [True, True, False],
                         "observed_arrivals_total": [0.0, 10.0, 999999.0],
                         "observed_nights_total": [0.0, 20.0, 999999.0]})
    scores = get_scores(rows, np.array([[2.0, 4.0], [10.0, 20.0], [0.0, 0.0]]))
    assert scores["arrivals_total"]["n"] == 2
    assert scores["arrivals_total"]["wape"] == 20
    assert scores["nights_total"]["wape"] == 20


def test_native_baseline_and_scaling_ignore_future_outcomes() -> None:
    """Exercise the real forecaster's fitted baselines, not only a dummy interface."""
    dates = pd.date_range("2019-01-01", periods=48, freq="MS")
    frame = pd.DataFrame({"date": dates, "geo": "AA", "accommodation_type": "total",
                         "arrivals_total": np.arange(48) + 1.0, "nights_total": np.arange(48) + 2.0,
                         "month": dates.month, "macro": np.arange(48), "regime_state": 0})
    model = RAExoDLinearForecaster(epochs=0, future_cols=["month"], past_exo_cols=["macro"],
                                   use_film=False, use_exo=False, use_consistency=False)
    model.device = torch.device("cpu")
    model.fit(frame.loc[frame.date.lt("2022-01-01")])
    rows = frame.loc[frame.date.ge("2022-01-01")].reset_index(drop=True)
    rows["origin"] = pd.Timestamp("2021-12-31 23:59:59")
    original, _ = build_core_inputs(model, frame, rows)
    changed = frame.copy()
    changed.loc[changed.date.ge("2022-01-01"), ["arrivals_total", "nights_total", "macro"]] = 1e15
    changed_rows = rows.copy()
    changed_rows[["arrivals_total", "nights_total", "macro"]] = 1e15
    alternative, _ = build_core_inputs(model, changed, changed_rows)
    for a, b in zip(original, alternative):
        assert torch.equal(a, b)


def test_projection_preserves_arrivals_and_satisfies_lower_bound() -> None:
    """Physical projection must not be mistaken for an arrivals accuracy change."""
    original = np.array([[0.0, -1.0], [20.0, 10.0], [5.0, 30.0]])
    result = project(original)
    np.testing.assert_array_equal(result[:, 0], original[:, 0])
    assert (result[:, 1] >= result[:, 0]).all()


def test_gate_falls_back_to_temporal_when_spatial_has_no_gain() -> None:
    """Endpoint fallback is required even when the scalar optimizer is interior."""
    rows = pd.DataFrame({"observed_pair": [True, True],
                         "observed_arrivals_total": [10.0, 20.0],
                         "observed_nights_total": [20.0, 40.0]})
    temporal = np.array([[10.0, 20.0], [20.0, 40.0]])
    assert choose_weight(rows, temporal, temporal + 100) == 1.0


def test_resume_refuses_different_source_identity() -> None:
    """An old prediction directory must never silently enter a new experiment."""
    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        prepare_directory(path, {"data_sha256": "first", "source_hashes": {}})
        with unittest.TestCase().assertRaisesRegex(ValueError, "Run identity changed"):
            prepare_directory(path, {"data_sha256": "different", "source_hashes": {}})


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    """Support the standard-library runner as well as pytest discovery."""
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))
