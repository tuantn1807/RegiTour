"""
Unit tests for the ExecutiveReportGenerator service.
"""

from pathlib import Path
import unittest

from src.services.report_generator import ExecutiveReportGenerator

ROOT_DIR = Path(__file__).resolve().parents[1]


class ExecutiveReportGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = ExecutiveReportGenerator(root_dir=ROOT_DIR)

    def test_generate_report_dict_structure(self):
        data = self.generator.generate_report_dict(geo="FR", accom_type="total", horizon_months=12)
        self.assertIn("metadata", data)
        self.assertIn("kpis", data)
        self.assertIn("invariant_audit", data)
        self.assertIn("workforce_capacity", data)
        self.assertIn("monthly_schedule", data)
        self.assertEqual(data["metadata"]["geo"], "FR")
        self.assertEqual(data["metadata"]["country"], "France")
        self.assertEqual(len(data["monthly_schedule"]), 12)
        self.assertEqual(data["invariant_audit"]["post_projection_violations"], 0)

    def test_generate_markdown_report_content(self):
        md = self.generator.generate_markdown_report(geo="DK", accom_type="total", horizon_months=6)
        self.assertIn("# 🏛️ Báo Cáo Điều Hành DMO — Denmark (DK)", md)
        self.assertIn("Tóm Lược Chỉ Số Điều Hành Cốt Lõi", md)
        self.assertIn("Kiểm Toán Nhất Quán Toán Học & Bất Biến Pi_C", md)
        self.assertIn("Lịch Trình Công Suất Lưu Trú & Nhân Sự Thời Vụ", md)
        self.assertIn("Khuyến Nghị Chiến Lược Cho Lãnh Đạo DMO", md)


    def test_generate_government_submission(self):
        sub = self.generator.generate_government_submission(geo="DK", horizon_months=12)
        self.assertIn("TỜ TRÌNH", sub)
        self.assertIn("PHƯƠNG ÁN PHỐI HỢP LIÊN NGÀNH", sub)
        self.assertIn("Chỉ tiêu Lượt khách", sub)
        self.assertIn("SỞ DU LỊCH", sub)


if __name__ == "__main__":
    unittest.main()

