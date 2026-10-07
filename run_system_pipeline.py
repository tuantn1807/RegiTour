"""Run the standalone annual World Bank tourist-arrivals training pipeline."""

from src.pipeline.worldbank_pipeline import run_world_bank_pipeline


def main():
    run_world_bank_pipeline()


if __name__ == "__main__":
    main()
