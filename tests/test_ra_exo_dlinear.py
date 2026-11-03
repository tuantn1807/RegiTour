"""Equation and integration regressions for the paper's DLinear forecast path."""

import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.baselines import VanillaDLinearForecaster
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.ra_exo_dlinear import (
    CoupledConsistencyLinearLoss,
    RAExoDLinearForecaster,
    RAExoDLinearNet,
    SeriesDecomp,
    VanillaDLinearNet,
)
from src.pipeline.linear_pipeline import TourismLinearForecastingPipeline


torch.set_num_threads(1)


def panel(annual: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=12 if annual else 36, freq="YS" if annual else "MS")
    rows = []
    for geo, scale in (("AA", 1.0), ("BB", 2.0)):
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "geo": geo, "accommodation_type": "total",
                "month": date.month, "year": date.year,
                "arrivals_total": scale * (100.0 + i * 3.0),
                "nights_total": scale * (210.0 + i * 4.0),
                "covid_stringency_index": 90.0 if date.year == 2020 else 0.0,
                "regime_state": i % 3,
            })
    return pd.DataFrame(rows)


class ForecastEquationTests(unittest.TestCase):
    def test_forward_matches_independent_numpy_equation(self):
        """Oracle includes moving average, affine branches, FiLM, and both streams."""
        torch.manual_seed(19)
        net = RAExoDLinearNet(lookback_len=5, forecast_horizon=2, past_exo_dim=2, future_exo_dim=2, kernel_size=3)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.uniform_(-0.3, 0.3)
        history = torch.rand(3, 5, 2) * 30
        future = torch.randn(3, 2, 2)
        past = torch.randn(3, 5, 2)
        regimes = torch.tensor([0, 1, 2])
        scales = [torch.tensor([[0.0, 0.5], [10.0, 2.0], [50.0, 100.0]])] * 2

        def affine(layer, values):
            return values @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()

        x = history.numpy()
        padded = np.pad(x, ((0, 0), (1, 1), (0, 0)), mode="edge")
        trend = np.stack([padded[:, i:i + 3].mean(axis=1) for i in range(5)], axis=1)
        seasonal = x - trend
        trend_projection = affine(net.linear_trend, trend.transpose(0, 2, 1)).transpose(0, 2, 1)
        seasonal_projection = affine(net.linear_seas, seasonal.transpose(0, 2, 1)).transpose(0, 2, 1)
        embedding = net.regime_embedding.weight.detach().numpy()[regimes.numpy()]
        expected = []
        for channel, suffix in enumerate(("a", "n")):
            gain = getattr(net, f"alpha_trend_{suffix}").item() * np.tanh(affine(getattr(net, f"film_trend_{suffix}"), embedding))
            beta = affine(getattr(net, f"film_bias_{suffix}"), embedding)
            temporal = (1 + gain) * trend_projection[:, :, channel] + beta + seasonal_projection[:, :, channel]
            delta_future = getattr(net, f"alpha_fut_{suffix}").item() * np.tanh(affine(getattr(net, f"linear_fut_{suffix}"), future.numpy()).squeeze(-1))
            delta_past = getattr(net, f"alpha_past_{suffix}").item() * np.tanh(affine(getattr(net, f"linear_past_{suffix}"), past.numpy().reshape(3, -1)))
            expected.append(np.maximum(temporal + np.maximum(scales[channel].numpy(), 1) * (delta_future + delta_past), 0))

        actual = net(history, future, past, regimes, *scales, return_components=True)
        for channel, name in enumerate(("arrivals", "nights")):
            np.testing.assert_allclose(actual[name].detach().numpy(), expected[channel], rtol=2e-5, atol=2e-5)
            factor = actual[f"baseline_{name}"] * (
                1 + actual[f"delta_trend_{name}"] + actual[f"delta_future_{name}"] + actual[f"delta_past_{name}"]
            )
            torch.testing.assert_close(actual[name], F.relu(factor), rtol=2e-5, atol=2e-5)

    def test_disabled_extensions_recover_dlinear_for_any_baseline(self):
        torch.manual_seed(3)
        model = RAExoDLinearNet(lookback_len=4, forecast_horizon=2, past_exo_dim=1, future_exo_dim=1, kernel_size=3, use_film=False, use_exo=False)
        vanilla = VanillaDLinearNet(lookback_len=4, forecast_horizon=2, kernel_size=3)
        vanilla.linear_trend.load_state_dict(model.linear_trend.state_dict())
        vanilla.linear_seas.load_state_dict(model.linear_seas.state_dict())
        history = torch.rand(3, 4, 2) * 10
        expected = F.relu(vanilla(history))
        for baseline in (0.0, 1.0, 100000.0):
            output = model(history, torch.zeros(3, 2, 1), torch.zeros(3, 4, 1), torch.zeros(3, dtype=torch.long), torch.full((3, 2), baseline), torch.full((3, 2), baseline))
            torch.testing.assert_close(torch.stack(output, dim=-1), expected)

    def test_raw_violation_survives_forward_and_penalty_has_gradient(self):
        net = RAExoDLinearNet(lookback_len=4, forecast_horizon=1, past_exo_dim=1, future_exo_dim=1, kernel_size=1, use_film=False, use_exo=False)
        history = torch.tensor([[[100.0, 10.0]] * 4])
        scale = torch.ones(1, 1)
        arrivals, nights = net(history, torch.zeros(1, 1, 1), torch.zeros(1, 4, 1), torch.tensor([0]), scale, scale)
        self.assertGreater(arrivals.item(), nights.item())
        arrivals.retain_grad()
        nights.retain_grad()
        loss = CoupledConsistencyLinearLoss(lambda_cons=0.2)(arrivals, arrivals.detach(), nights, nights.detach(), scale, scale)
        loss.backward()
        self.assertGreater(arrivals.grad.item(), 0)
        self.assertLess(nights.grad.item(), 0)

    def test_loss_matches_normalized_mse_and_consistency_equation(self):
        a = torch.tensor([[10.0, 4.0]])
        n = torch.tensor([[2.0, 8.0]])
        sa, sn = torch.tensor([[2.0, 2.0]]), torch.tensor([[4.0, 4.0]])
        ya, yn = torch.tensor([[8.0, 2.0]]), torch.tensor([[6.0, 8.0]])
        # Prediction MSE = 1 + 0.5; physical penalty mean = (4**2 + 0)/2.
        actual = CoupledConsistencyLinearLoss(lambda_cons=0.2)(a, ya, n, yn, sa, sn)
        self.assertAlmostEqual(actual.item(), 1.5 + 0.2 * 8.0, places=6)
        without = CoupledConsistencyLinearLoss(lambda_cons=0)(a, ya, n, yn, sa, sn)
        self.assertAlmostEqual(without.item(), 1.5, places=6)

    def test_physical_projection_preserves_anchor_and_does_not_enforce_economic_band(self):
        arrivals = np.array([0.0, 0.25, 20.0, 10.0], dtype=np.float32)
        nights = np.array([0.0, 0.1, 500.0, 3.0], dtype=np.float32)
        out_a, out_n, flags = CoupledConsistencyOptimizer.project_physical_pair(arrivals, nights)
        np.testing.assert_array_equal(out_a, arrivals)
        np.testing.assert_allclose(out_n, [0.0, 0.25, 500.0, 10.0])
        np.testing.assert_array_equal(flags, [False, True, False, True])

    def test_kernel_rejects_even_and_nonpositive_values(self):
        for kernel in (0, -1, 2, 4):
            with self.assertRaises(ValueError):
                SeriesDecomp(kernel)

    def test_hmm_state_takes_precedence_over_calendar_regime(self):
        frame = pd.DataFrame({"regime_state": [2, 1], "period_group": ["pre_2020", "from_2022"]})
        np.testing.assert_array_equal(RAExoDLinearForecaster._encode_regimes(frame), [2, 1])

    def test_fusion_weight_uses_physical_projection_and_considers_endpoints(self):
        pipeline = object.__new__(TourismLinearForecastingPipeline)
        pipeline.data_loader = SimpleNamespace(target_arrivals="arrivals_total", target_nights="nights_total")
        pipeline.coupled_optimizer = CoupledConsistencyOptimizer()
        validation = pd.DataFrame({"arrivals_total": [1.0, 4.0], "nights_total": [20.0, 4.0]})
        good_a, good_n = np.array([1.0, 4.0]), np.array([20.0, 0.0])
        bad_a = bad_n = np.array([100.0, 100.0])
        for temporal_good, expected in ((True, 1.0), (False, 0.0)):
            with redirect_stdout(StringIO()):
                weight = pipeline._optimize_fusion_weight(
                    validation,
                    good_a if temporal_good else bad_a,
                    good_n if temporal_good else bad_n,
                    bad_a if temporal_good else good_a,
                    bad_n if temporal_good else good_n,
                )
            self.assertEqual(weight, expected)


class ForecasterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = panel(annual=True)
        cls.train = frame[frame.year < 2019].copy()
        cls.test = frame[frame.year >= 2019].copy()
        cls.model = RAExoDLinearForecaster(lookback_len=4, forecast_horizon=1, epochs=2, kernel_size=3, future_cols=[], past_exo_cols=["covid_stringency_index"])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element tensors")
            cls.model.fit(cls.train)

    def test_prediction_equals_exposed_components_without_extra_postprocessing(self):
        arrivals, nights = self.model.predict(self.test, history_df=self.train)
        components = self.model.decompose(self.test, history_df=self.train)
        np.testing.assert_allclose(arrivals, components["arrivals_raw"])
        np.testing.assert_allclose(nights, components["nights_raw"])
        for name in ("arrivals", "nights"):
            reconstructed = components[f"temporal_{name}"] + components[f"future_contribution_{name}"] + components[f"past_contribution_{name}"] + components[f"nonnegative_adjustment_{name}"]
            np.testing.assert_allclose(components[name], reconstructed, rtol=2e-5, atol=2e-5)

    def test_current_year_policy_does_not_apply_an_undocumented_2020_multiplier(self):
        current = self.test[self.test.year == 2020].copy()
        history = pd.concat([self.train, self.test[self.test.year == 2019]], ignore_index=True)
        changed = current.copy()
        changed["covid_stringency_index"] = 0.0
        for actual, expected in zip(self.model.predict(current, history), self.model.predict(changed, history)):
            np.testing.assert_array_equal(actual, expected)

    def test_annual_scale_is_training_median(self):
        future = self.test[self.test.geo == "AA"]
        a, n = self.model._baseline_sequence("AA", "total", future)
        np.testing.assert_array_equal(a, np.full(len(future), self.train[self.train.geo == "AA"].arrivals_total.median()))
        np.testing.assert_array_equal(n, np.full(len(future), self.train[self.train.geo == "AA"].nights_total.median()))

    def test_missing_entity_history_raises_instead_of_fabricating_a_baseline(self):
        unknown = self.test.iloc[:1].copy()
        unknown["geo"] = "ZZ"
        with self.assertRaises(ValueError):
            self.model.predict(unknown, history_df=self.train)

    def test_zero_output_is_not_raised_to_ten_or_a_historical_los_ratio(self):
        # Use an independent model to preserve the shared fit above.
        model = RAExoDLinearForecaster(lookback_len=4, forecast_horizon=1, epochs=0, kernel_size=3, use_exo=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element tensors")
            model.fit(self.train)
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.zero_()
        a, n = model.predict(self.test, history_df=self.train)
        np.testing.assert_array_equal(a, np.zeros(len(self.test)))
        np.testing.assert_array_equal(n, np.zeros(len(self.test)))

    def test_averaged_components_reconstruct_forecasts_when_relu_is_active(self):
        frame = panel()
        train = frame[frame.date < "2012-07-01"].copy()
        test = frame[frame.date >= "2012-07-01"].copy()
        model = VanillaDLinearForecaster(lookback_len=12, forecast_horizon=3, kernel_size=3, epochs=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element tensors")
            model.fit(train)
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.zero_()
            model.net.linear_trend.weight[:, -1] = 1.0
            model.net.linear_trend.bias.copy_(torch.tensor([-200.0, 0.0, -50.0]))
        a, n = model.predict(test, history_df=train)
        components = model.decompose(test, history_df=train)
        self.assertTrue(np.any(components["nonnegative_adjustment_arrivals"] > 0))
        for name, actual in (("arrivals", a), ("nights", n)):
            reconstructed = components[f"temporal_{name}"] + components[f"nonnegative_adjustment_{name}"]
            np.testing.assert_allclose(reconstructed, actual, rtol=2e-5, atol=2e-5)

    def test_vanilla_baseline_uses_target_history_and_exact_dlinear(self):
        frame = panel()
        train = frame[frame.date < "2012-07-01"].copy()
        test = frame[frame.date >= "2012-07-01"].copy()
        model = VanillaDLinearForecaster(lookback_len=12, forecast_horizon=3, kernel_size=3, epochs=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element tensors")
            model.fit(train)
        a, n = model.predict(test, history_df=train)
        self.assertEqual(a.shape, (len(test),))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(n).all())
        self.assertFalse(model.use_film or model.use_exo or model.use_consistency)
        components = model.decompose(test, history_df=train)
        for name in ("arrivals", "nights"):
            expected = np.maximum(components[f"dlinear_trend_{name}"] + components[f"dlinear_seasonal_{name}"], 0)
            np.testing.assert_allclose(components[name], expected, rtol=2e-5)


class ShortHistoryForecastTests(unittest.TestCase):
    def setUp(self):
        frame = panel(annual=True)
        long_history = frame[(frame.geo == "AA") & (frame.year < 2019)].copy()
        parts = [long_history]
        targets = []
        for geo, count in (("ARE", 1), ("MDG", 2), ("SVK", 3)):
            short = long_history.tail(count).copy()
            short["geo"] = geo
            parts.append(short)
            target = frame[(frame.geo == "AA") & (frame.year == 2019)].copy()
            target["geo"] = geo
            targets.append(target)
        self.train = pd.concat(parts, ignore_index=True)
        self.test = pd.concat(targets, ignore_index=True)
        self.model = RAExoDLinearForecaster(
            lookback_len=4, forecast_horizon=1, kernel_size=3, epochs=0,
            future_cols=["month"], past_exo_cols=["covid_stringency_index"],
            use_film=False, use_exo=False,
        ).fit(self.train)

    def test_one_two_and_three_observations_are_padded_using_only_past_values(self):
        windows = self.model._make_forecast_windows(self.test, history_df=self.train)
        np.testing.assert_array_equal(windows["history_observations"], [1, 2, 3])
        for i, geo in enumerate(("ARE", "MDG", "SVK")):
            values = self.train[self.train.geo == geo][["arrivals_total", "nights_total"]].to_numpy()
            expected = np.concatenate([np.repeat(values[:1], 4 - len(values), axis=0), values])
            np.testing.assert_array_equal(windows["history"][i], expected)
        self.assertEqual(len(self.model._fit_frame), len(self.train))

    def test_padded_predictions_use_the_model_and_publish_diagnostics(self):
        with self.assertWarnsRegex(UserWarning, "3 forecast rows use short histories"):
            arrivals, nights = self.model.predict(self.test, history_df=self.train)
        components = self.model.decompose(self.test, history_df=self.train)
        np.testing.assert_allclose(arrivals, components["arrivals_raw"])
        np.testing.assert_allclose(nights, components["nights_raw"])
        diagnostics = self.model.last_prediction_diagnostics
        np.testing.assert_array_equal(diagnostics["history_observations_min"], [1, 2, 3])
        np.testing.assert_array_equal(diagnostics["history_padding_max"], [3, 2, 1])
        self.assertTrue(np.isfinite(arrivals).all() and np.isfinite(nights).all())

    def test_changing_unobserved_targets_does_not_change_padding_or_forecast(self):
        changed = self.test.copy()
        changed[["arrivals_total", "nights_total"]] = 10000000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = self.model.predict(self.test, history_df=self.train)
            actual = self.model.predict(changed, history_df=self.train)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_strict_policy_rejects_short_history(self):
        self.model.short_history_policy = "error"
        with self.assertRaises(ValueError):
            self.model.predict(self.test, history_df=self.train)

    def test_zero_past_observations_still_raise_with_entity_details(self):
        unknown = self.test.iloc[:1].copy()
        unknown["geo"] = "ZZ"
        target = pd.concat([self.test, unknown], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "ZZ"):
            self.model.predict(target, history_df=self.train)

    def test_padding_supports_multiple_horizons_and_empty_exogenous_streams(self):
        model = RAExoDLinearForecaster(
            lookback_len=4, forecast_horizon=3, kernel_size=3, epochs=0,
            future_cols=[], past_exo_cols=[], use_film=False, use_exo=False,
        )
        target = panel(annual=True)
        target = target[(target.geo == "AA") & (target.year >= 2019)].copy()
        target["geo"] = "ARE"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(self.train)
            arrivals, nights = model.predict(target, history_df=self.train)
        self.assertEqual(len(arrivals), 3)
        self.assertTrue(np.isfinite(arrivals).all() and np.isfinite(nights).all())
        np.testing.assert_array_equal(model.last_prediction_diagnostics["history_padding_max"], [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
