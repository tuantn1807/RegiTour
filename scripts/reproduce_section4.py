"""Reproduce the matched Section 4 experiment, diagnostics, tables, and manuscript."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.section4_experiments import OUTPUT, file_hash, write_json


def run_script(name: str, *arguments: str) -> None:
    """Stream one reproducible stage and stop on its first failure."""
    subprocess.run([sys.executable, "-u", "-X", "utf8", str(ROOT / "scripts" / name), *arguments], cwd=ROOT, check=True)


def write_package_record(output: Path) -> None:
    """Archive exact inputs and code and map every requested checklist item to evidence."""
    if not (output / "summary.json").exists():
        raise FileNotFoundError("Verified summary.json is required before recording checklist completion")
    sources = ["scripts/section4_experiments.py", "scripts/section4_extra_baselines.py",
               "scripts/analyze_section4.py", "scripts/revise_section4_manuscript.py",
               "scripts/reproduce_section4.py", "scripts/tune_ets_baseline.py",
               "scripts/reviewer_experiments.py", "src/data/dataset_loader.py",
               "src/models/ra_exo_dlinear.py", "src/models/trmf_spatial.py",
               "src/models/transformer_baselines.py", "src/models/coupled_optimizer.py",
               "src/models/dcqf_conformal.py", "tests/test_section4_protocol.py"]
    inputs = ["src/data/tourism_europe_enriched.csv", "data/processed/tourism_un_tourism_enriched.csv",
              "data_oced/tourism_economics_worldbank-3.xlsx"]
    artifacts = ["paper/paper_en.tex", "paper/paper_en.pdf", "paper/tables/section4_numbers.tex",
                 "paper/tables/section4_benchmark.tex", "paper/tables/section4_regimes.tex",
                 "paper/tables/section4_components_global.tex", "paper/figures/section4_country_cases.png",
                 "paper/figures/section4_architecture.pdf"]
    for relative in sources + inputs + artifacts:
        destination = output / "reproduction_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    package = {"created_on": "2026-11-27", "python": platform.python_version(),
               "python_executable": sys.executable, "processor": platform.processor(),
               "logical_cpus": os.cpu_count(), "packages": {p: importlib.metadata.version(p) for p in
               ["numpy", "pandas", "torch", "scipy", "hmmlearn", "statsmodels", "lightgbm", "catboost", "scikit-learn", "matplotlib", "openpyxl"]},
               "source_hashes": {p: file_hash(ROOT / p) for p in sources},
               "input_hashes": {p: file_hash(ROOT / p) for p in inputs},
               "artifact_hashes": {p: file_hash(ROOT / p) for p in artifacts},
               "limitations": ["No historical publication vintages; retrospective reference-period evaluation",
                               "Monthly source-stage HICP imputation cannot be fully undone",
                               "Annual nights mostly constructed; source-specific errors reported",
                               "Main results use one seed and finite tuning budgets; bootstrap resamples destinations"]}
    write_json(package, output / "reproduction_manifest.json")
    descriptions = {
        "4.1.1": "Audited source keys, paired masks, zero counts, target scope, missing data, annual source flags, chronological splits and calendar-year annual grids.",
        "4.1.2": "Documented January origins, pre-origin histories, fixed parameter boundaries, prefix-filtered HMM, publication/vintage limitations, and upstream filling risks; tested future-outcome invariance.",
        "4.1.3": "Retained exact input/code snapshots, hashes, settings, checkpoints, keyed predictions, masks, environment, timings and a single reproduction command.",
        "4.2.1": "Table 2 combines matched accuracy with active parameters and measured CPU training-plus-forecast cost.",
        "4.2.2": "Recomputed WAPE/MAE from keyed predictions; verified every CSV checksum and corresponding JSON metric.",
        "4.2.3": "Generated shared numeric macros and replaced stale accuracy, efficiency and significance claims.",
        "4.3.1": "Table 3 separates temporal references, TRMF-only, fixed 0.7 fusion and bounded validation-selected adaptive fusion.",
        "4.3.2": "Traced archived 66.69% to train<=2017, validation 2018, calibration 2019, test 2020-2021, regularized sigmoid gate and ungated FiLM.",
        "4.3.3": "Distinguished the legacy sigmoid weight 0.7380328366 from the fixed static prior and newly rerun adaptive full model.",
        "4.3.4": "Ran Seasonal Naive on all four stress folds with identical eligible keys.",
        "4.3.5": "Ran RA-Exo temporal control and TRMF-only on all four folds; also added genuinely trained DLinear.",
        "4.3.6": "Reported a separate validation-selected temporal weight for every fold.",
        "4.4.1": "Panel 4A isolates exogenous inputs, HMM intercept, ungated FiLM, soft gating, coupled loss, spatial fusion and final projection.",
        "4.4.2": "Rejected the unsupported equality of Full and Exogenous-only metrics using their freshly generated forecast arrays.",
        "4.4.3": "Mapped each ablation to explicit implementation flags; distinguished frozen and learned DLinear and approximate TRMF implementation.",
        "4.4.4": "Separated cross-fold temporal robustness in Table 3 from recent-fold component contributions in Panel 4A.",
        "4.5.1": "Panel 4B reports new annual re-estimation against Persistence, temporal references and TRMF-only; does not claim zero-shot transfer.",
        "4.5.2": "Preserved observed annual outcomes, rebuilt calendar-year macro lags from raw WDI without backward fill, and scored common positive-pair masks with source-specific nights subsets.",
        "4.5.3": "Replaced universal superiority claims with overall and period-specific results, coverage changes and synthetic-target limitations.",
        "4.6.1": "Regenerated Figure 2 from the new keyed forecasts; measured pre/post projection violations, LOS errors, undefined denominators and numerical rounding.",
        "4.6.2": "Recalculated empirical marginal coverage and normalized width using separate held-out calibration; stated temporal and joint-coverage limitations.",
        "4.6.3": "Replaced outcome-trained legacy SHAP with held-out forecast-surrogate SHAP and measured fidelity; reran actual network HICP perturbations instead of fixed-elasticity simulations.",
        "4.6.4": "Used saved forecasts and generated numbers throughout; removed unsupported causal elasticity, guaranteed coverage and generic operational-feasibility claims.",
    }
    write_json({key: {"status": "completed", "work": value,
                     "evidence": ["summary.json", "reproduction_manifest.json", "paper/paper_en.tex"]}
                for key, value in descriptions.items()}, output / "checklist_status.json")
    readme = """# Section 4 reproducibility package

This package supports the six requested manuscript subsections and all 23 items
in the supplied work checklist. Earlier result directories and user-edited source
data are preserved. `checklist_status.json` maps every item to completed work;
the completion is an audit/revision status, not a claim that scientific limitations
such as unavailable publication vintages have disappeared.

From the repository root, with the recorded Python environment:

```powershell
python -u -X utf8 scripts/reproduce_section4.py --output results/section4_reproduction
```

Use a fresh output directory for independent reproduction. An interrupted run can
resume only with matching source/input identities. The command writes the new
tables, figures, numeric macros and manuscript source. Existing model outputs in
the selected directory are reused; a fresh directory forces training again.

The stages are monthly matched experiments; annual re-estimation; validation
selection of the primary scaled-count ETS baseline; trainable DLinear controls;
direct HICP input interventions and held-out forecast-surrogate TreeSHAP;
key/mask/origin/checksum/metric verification; and manuscript regeneration.
The legacy `run_un_tourism_dlinear.py` remains a temporal adaptation. The annual
results in this manuscript use `section4_experiments.py --stage annual`, which
also includes prefix-filtered HMM states, annual spatial factors and validation
fusion. These entry points should not be treated as interchangeable runs.

`reproduction_snapshot` preserves exact inputs and implementations. Each fold
also retains a manifest, completed histories, forecast rows, model checkpoints,
and keyed CSV/JSON results. `summary.json` independently recomputes every main
metric. CPU timings include feature preparation and are hardware-dependent.
`parameter_counts.csv` distinguishes active forecast-graph parameters from
registered parameters in disabled wrapper branches.

The 66.69% pandemic result is historical: a regularized sigmoid gate with weight
0.7380328366, ungated FiLM and the 2020-2021 test fold. It is retained as an
explicitly labelled archived row, not substituted for the new static 0.7 blend or
the bounded adaptive model. Earlier 9.16%/9.64% full-model manuscript claims are replaced
by the new keyed run. Exogenous-only and full forecasts are compared directly.

The old four-feature SHAP script fitted observed arrivals, rather than the current
forecaster's predictions. The former -7.20%/-7.56% price response used a fixed
-0.72 elasticity. Those numbers are not model estimates. This revision instead
trains a surrogate on 2022-2023 model forecasts, evaluates 2024-2026 fidelity,
reports SHAP in log-output units, and forwards perturbed HICP histories through
the actual stored model. Forecast associations and interventions are not causal
tourism elasticities.

Main limitations remain explicit: retrospective reference-period availability,
unavailable historical release vintages, monthly upstream HICP filling,
constructed annual nights and unequal later-year country coverage, single-seed
model fits, finite tuning budgets, and approximate TRMF factor optimization.
Separate marginal conformal intervals do not guarantee joint coverage or joint
cross-target feasibility. The observed Eurostat snapshot itself has 14 N<A pairs,
including two recent test pairs; these are retained in the common scoring mask.

Protocol invariants can be checked with:

```powershell
python -m unittest discover -s tests -p test_section4_protocol.py -v
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / ".gitignore").write_text("# Generated model state remains available locally.\n**/*.pt\n**/*.pkl\n**/*.cbm\n", encoding="utf-8")


def main() -> None:
    """Run every stage, or finalize the evidence record for completed stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.record_only:
        run_script("section4_experiments.py", "--stage", "monthly", "--fold", "all", "--output", str(output))
        run_script("section4_experiments.py", "--stage", "annual", "--output", str(output))
        options = ["--output", str(output)]
        ets = output / "monthly/recent/ets.json"
        if ets.exists() and json.loads(ets.read_text()).get("selection"):
            options.append("--skip-ets")
        run_script("section4_extra_baselines.py", *options)
        run_script("analyze_section4.py", "--stage", "all", "--output", str(output))
        run_script("revise_section4_manuscript.py")
    write_package_record(output)


if __name__ == "__main__":
    main()
