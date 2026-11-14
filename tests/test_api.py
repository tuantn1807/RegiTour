"""
Unit and integration tests for the RegiTour REST API endpoints.
"""

from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class RegiTourAPITests(unittest.TestCase):
    def test_root_endpoint_returns_online(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("documentation", data)

    def test_health_endpoint(self):
        response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["total_destinations"], 38)
        self.assertIn("2023-12-31", data["available_origins"])

    def test_destinations_endpoint(self):
        response = client.get("/api/v1/destinations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total"], 38)
        geos = [d["geo"] for d in data["destinations"]]
        self.assertIn("FR", geos)
        self.assertIn("CY", geos)
        self.assertIn("SE", geos)
        self.assertIn("DK", geos)
        self.assertIn("hotel", data["nace_categories"])

    def test_forecast_plan_endpoint_success(self):
        payload = {
            "geo": "FR",
            "accommodation_type": "total",
            "horizon_months": 12,
            "origin_label": "2023-12-31",
        }
        response = client.post("/api/v1/forecast/plan", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["geo"], "FR")
        self.assertEqual(len(data["monthly_plan"]), 12)
        summary = data["summary"]
        self.assertEqual(summary["post_violations"], 0)
        self.assertGreater(summary["total_arrivals"], 0)
        self.assertGreater(summary["total_nights"], 0)
        self.assertGreaterEqual(summary["avg_los"], 1.0)

    def test_forecast_plan_endpoint_not_found(self):
        payload = {
            "geo": "NON_EXISTENT_GEO",
            "accommodation_type": "total",
            "horizon_months": 12,
        }
        response = client.post("/api/v1/forecast/plan", json=payload)
        self.assertEqual(response.status_code, 404)

    def test_capacity_staffing_endpoint_success(self):
        payload = {
            "geo": "FR",
            "horizon_months": 12,
            "origin_label": "2023-12-31",
            "hours_per_arrival": 0.5,
            "hours_per_night": 0.35,
            "hours_per_fte_month": 160.0,
            "maintenance_occ_threshold_pct": 35.0,
        }
        response = client.post("/api/v1/capacity/staffing", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["geo"], "FR")
        summary = data["summary"]
        self.assertGreater(summary["total_peak_fte"], summary["total_baseline_fte"])
        self.assertIn("hotel", summary["categories"])
        self.assertIn("short_stay", summary["categories"])
        self.assertIn("camping", summary["categories"])
        self.assertEqual(len(data["monthly_breakdown"]), 36)

    def test_stress_test_simulation_endpoint(self):
        payload = {
            "geo": "CY",
            "accommodation_type": "total",
            "horizon_months": 12,
            "origin_label": "2023-12-31",
            "hicp_shock_pct": 20.0,
            "stringency_delta": 30.0,
        }
        response = client.post("/api/v1/simulation/stress-test", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["geo"], "CY")
        self.assertEqual(data["post_pi_c_violations"], 0)
        self.assertLess(data["arrivals_change_pct"], 0.0)
        self.assertEqual(len(data["monthly_impact"]), 12)

    def test_market_regime_timeline_endpoint(self):
        response = client.get("/api/v1/market-regime/timeline")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total_observations"], 0)
        self.assertIn(data["latest_state"], [0, 1, 2])
        self.assertTrue(len(data["timeline"]) > 0)

    def test_recovery_benchmark_endpoint(self):
        response = client.get("/api/v1/recovery/benchmark?horizon_months=12&origin_label=2023-12-31")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_countries"], 38)
        geos = [b["geo"] for b in data["benchmarks"]]
        self.assertIn("CY", geos)
        self.assertIn("SE", geos)
        self.assertIn("DK", geos)

    def test_pi_c_optimizer_endpoint(self):
        # Introduce physical violation: Nights (50, 80) < Arrivals (100, 100)
        payload = {
            "arrivals": [100.0, 100.0, 50.0],
            "nights": [50.0, 80.0, 150.0],
        }
        response = client.post("/api/v1/optimizer/pi-c", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["violations_detected"], 2)
        self.assertTrue(data["all_constraints_satisfied"])
        # Nights must have been adjusted up to at least arrivals
        self.assertGreaterEqual(data["projected_nights"][0], 100.0)
        self.assertGreaterEqual(data["projected_nights"][1], 100.0)
        self.assertEqual(data["projected_nights"][2], 150.0)
        for los in data["projected_los"]:
            self.assertGreaterEqual(los, 1.0)

    def test_report_generation_endpoint_markdown_and_json(self):
        # Markdown
        req_md = {"geo": "FR", "accommodation_type": "total", "horizon_months": 12, "format": "markdown"}
        res_md = client.post("/api/v1/reports/generate", json=req_md)
        self.assertEqual(res_md.status_code, 200)
        data_md = res_md.json()
        self.assertIn("# 🏛️ Báo Cáo Điều Hành DMO", data_md["content"])

        # JSON
        req_json = {"geo": "FR", "accommodation_type": "total", "horizon_months": 12, "format": "json"}
        res_json = client.post("/api/v1/reports/generate", json=req_json)
        self.assertEqual(res_json.status_code, 200)
        data_json = res_json.json()
        self.assertIn("kpis", data_json["content"])


if __name__ == "__main__":
    unittest.main()

