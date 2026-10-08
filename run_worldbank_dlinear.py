"""
Script: run_worldbank_dlinear.py
Mục đích: Thực thi độc lập framework RA-HE-DLinear trên tập dữ liệu thường niên World Bank / OECD
          phục vụ đối sánh đa tập dữ liệu cho đề tài NCKH.
Cách dùng:
    python run_worldbank_dlinear.py
"""

from src.pipeline.worldbank_dlinear_pipeline import WorldBankLinearForecastingPipeline


def main():
    pipeline = WorldBankLinearForecastingPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
