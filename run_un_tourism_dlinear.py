"""
Script: run_un_tourism_dlinear.py
Mục đích: Thực thi độc lập framework RA-HE-DLinear trên tập dữ liệu thường niên UN Tourism (1995 - 2024)
          phục vụ đối sánh đa tập dữ liệu và mở rộng trước, trong, sau COVID.
Cách dùng:
    python run_un_tourism_dlinear.py
"""

from src.pipeline.un_tourism_dlinear_pipeline import UNTourismLinearForecastingPipeline


def main():
    pipeline = UNTourismLinearForecastingPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
