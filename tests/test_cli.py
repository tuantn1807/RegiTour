"""
Unit tests for the RegiTour CLI commands.
"""

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cli import cmd_audit, cmd_plan, cmd_report, cmd_staffing, cmd_stress


class RegiTourCLITests(unittest.TestCase):
    def test_cli_plan_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        class DummyArgs:
            geo = "FR"
            accom = "total"
            horizon = 6
            origin = "2023-12-31"
            output = temp_path

        cmd_plan(DummyArgs())

        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["geo"], "FR")
        self.assertIn("summary", data)
        self.assertEqual(len(data["monthly_plan"]), 6)
        Path(temp_path).unlink(missing_ok=True)

    def test_cli_staffing_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        class DummyArgs:
            geo = "FR"
            horizon = 6
            origin = "2023-12-31"
            output = temp_path

        cmd_staffing(DummyArgs())

        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["geo"], "FR")
        self.assertIn("summary", data)
        self.assertIn("hotel", data["summary"]["categories"])
        Path(temp_path).unlink(missing_ok=True)

    def test_cli_stress_stdout(self):
        class DummyArgs:
            geo = "CY"
            hicp = 15.0
            stringency = 20.0
            horizon = 6

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_stress(DummyArgs())
            output = fake_out.getvalue()

        self.assertIn("KẾT QUẢ STRESS TEST VĨ MÔ", output)
        self.assertIn("CY", output)
        self.assertIn("Số kỳ vi phạm Pi_C: 0", output)

    def test_cli_report_markdown_output(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            temp_path = tf.name

        class DummyArgs:
            geo = "SE"
            horizon = 6
            origin = "2023-12-31"
            format = "markdown"
            output = temp_path

        cmd_report(DummyArgs())

        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("# 🏛️ Báo Cáo Điều Hành DMO — Sweden (SE)", content)
        self.assertIn("Pi_C", content)
        Path(temp_path).unlink(missing_ok=True)

    def test_cli_audit_stdout(self):
        class DummyArgs:
            pass

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_audit(DummyArgs())
            output = fake_out.getvalue()

        self.assertIn("KẾT QUẢ KIỂM TOÁN", output)
        self.assertIn("Hoàn toàn 0 vi phạm", output)


if __name__ == "__main__":
    unittest.main()

