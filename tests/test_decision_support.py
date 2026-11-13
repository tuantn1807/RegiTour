"""Unit and UI smoke tests for the RegiTour DMO Operational Decision Support System."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.decision_support.dmo_engine import (
    NACE_LABELS,
    compute_coordinated_throughput_plan,
    compute_crisis_regime_diagnostics,
    compute_hmm_regime_timeline,
    compute_nace_capacity_staffing,
    compute_recovery_benchmarking,
    load_dss_bundle,
    simulate_macro_stress,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class DecisionSupportEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_dss_bundle(ROOT_DIR)

    def test_bundle_loads_eurostat_destinations_and_nace_categories(self):
        self.assertGreaterEqual(self.bundle.forecasts_df["geo"].nunique(), 38)
        self.assertTrue({"hotel", "short_stay", "camping", "total"}.issubset(
            set(self.bundle.forecasts_df["accommodation_type"].unique())
        ))
        self.assertIsNotNone(self.bundle.history_df)
        self.assertEqual(NACE_LABELS["hotel"][0], "I551")
        self.assertEqual(NACE_LABELS["short_stay"][0], "I552")
        self.assertEqual(NACE_LABELS["camping"][0], "I553")

    def test_layer1_coordinated_throughput_eliminates_physical_violations(self):
        plan_df, summary = compute_coordinated_throughput_plan(
            self.bundle.forecasts_df, geo="AL", accom_type="camping", horizon_months=12, origin_label="2023-12-31"
        )
        self.assertEqual(len(plan_df), 12)
        self.assertEqual(summary["post_violations"], 0)
        self.assertTrue((plan_df["pred_nights_safe"] >= plan_df["pred_arrivals_safe"]).all())
        self.assertGreater(summary["raw_violations"], 0)
        self.assertGreater(summary["total_nights_corrected"], 0.0)
        self.assertTrue((plan_df["pred_los"] >= 1.0).all())

    def test_layer2_nace_capacity_and_staffing_decomposition(self):
        cap_df, summary = compute_nace_capacity_staffing(
            self.bundle.forecasts_df,
            geo="FR",
            horizon_months=12,
            maintenance_occ_threshold_pct=40.0,
            origin_label="2023-12-31",
        )
        self.assertEqual(set(cap_df["nace_code"].unique()), {"I551", "I552", "I553"})
        self.assertEqual(len(cap_df), 36)
        self.assertGreater(summary["total_peak_fte"], summary["total_baseline_fte"])
        self.assertTrue((cap_df["required_fte"] >= 0.0).all())
        self.assertTrue((cap_df["occupancy_pct"] >= 0.0).all())

    def test_layer3_crisis_diagnostics_hmm_and_macro_stress_projection(self):
        diag = compute_crisis_regime_diagnostics(self.bundle, geo="CY")
        fold_df = diag["fold_comparison_df"]
        self.assertFalse(fold_df.empty)
        shock_row = fold_df[fold_df["fold"] == "shock"].iloc[0]
        self.assertLess(shock_row["ra_he_arr_wape"], shock_row["dlinear_arr_wape"])

        hmm_df = compute_hmm_regime_timeline(self.bundle.history_df)
        self.assertFalse(hmm_df.empty)
        prob_sums = hmm_df[["p_normal", "p_disruption", "p_recovery"]].sum(axis=1)
        np.testing.assert_allclose(prob_sums.to_numpy(), 1.0, atol=1e-4)

        plan_df, _ = compute_coordinated_throughput_plan(
            self.bundle.forecasts_df, geo="CY", accom_type="total", horizon_months=12, origin_label="2023-12-31"
        )
        stressed = simulate_macro_stress(
            plan_df, price_elasticity=diag["price_elasticity"], hicp_shock_pct=25.0, stringency_delta=50.0
        )
        self.assertTrue((stressed["stressed_nights"] >= stressed["stressed_arrivals"]).all())
        self.assertTrue((stressed["arrivals_delta_pct"] <= 0.0).all())

    def test_layer4_recovery_benchmarking_and_conformal_bounds(self):
        traj_df, cross_df = compute_recovery_benchmarking(
            self.bundle.forecasts_df,
            self.bundle.history_df,
            geo="DK",
            accom_type="total",
            horizon_months=12,
            origin_label="2023-12-31",
        )
        self.assertEqual(len(traj_df), 12)
        self.assertTrue((traj_df["upper_arrivals"] >= traj_df["pred_arrivals"]).all())
        self.assertTrue((traj_df["pred_arrivals"] >= traj_df["lower_arrivals"]).all())
        self.assertGreaterEqual(len(cross_df), 38)
        self.assertTrue({"CY", "SE", "DK"}.issubset(set(cross_df["geo"].unique())))


class DashboardAppSmokeTests(unittest.TestCase):
    def test_streamlit_app_renders_all_decision_layers_without_errors(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest unavailable")

        app_path = str(ROOT_DIR / "dashboard" / "app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, msg=f"Initial render failed: {at.exception}")

        nav_radio = at.sidebar.radio[0]
        for option in nav_radio.options:
            nav_radio.set_value(option).run()
            self.assertFalse(at.exception, msg=f"Section '{option}' raised exception: {at.exception}")


if __name__ == "__main__":
    unittest.main()

