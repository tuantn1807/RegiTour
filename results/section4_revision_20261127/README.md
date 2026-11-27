# Section 4 reproducibility package

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
