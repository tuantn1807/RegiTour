"""Run the standalone annual World Bank tourist-arrivals linear pipeline."""

from src.pipeline.worldbank_linear_pipeline import run_world_bank_linear_pipeline


def main():
    run_world_bank_linear_pipeline()


if __name__ == "__main__":
    main()
