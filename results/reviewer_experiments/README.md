# Eurostat baseline ablation and tourism experiments

These outputs support the revised baseline, ablation, and tourism sections of `paper/paper_en.tex`. The experiment uses the CSV snapshot at `src/data/tourism_europe_enriched.csv`, ending July 2026. All models are scored on the same originally observed, nonnegative arrivals/nights pairs. The recent test has 4,712 grid rows and 4,522 scored pairs.

Run the stages from the repository root in this order.

```powershell
python scripts/reviewer_experiments.py --fold all
python scripts/tune_ets_baseline.py
python scripts/analyze_reviewer_experiments.py
```

The first stage supports resuming completed model files with matching run metadata. The second stage supplies the primary ETS baseline on scaled counts with trend choice selected using 2022 validation outcomes. The fixed damped log-ETS and validation-selected log-ETS sensitivity outputs remain archived separately. The third stage creates summary CSV/JSON files, the LaTeX tables included by the paper, and the new country/LOS/projection figures.

Parameters are trained through 2021 in the recent fold. January origins predict up to 12 months, using historical observations before each origin. Validation is 2022 and residual calibration is 2023. Tranquil, shock, and recovery folds test 2018–2019, 2020–2021, and 2022–2023 with fitting boundaries moved back by six, four, and two years respectively. Models within a fold share target keys, origins, scoring eligibility, and the completed historical input frame. Neural/LightGBM output windows require complete observed labels. Statistical fits use the completed training series and record failed or constant-series seasonal fallbacks.

The HMM-only control adds a regime-dependent intercept. The FiLM-only control holds the state embedding constant. Joint HMM/FiLM enables state-dependent affine modulation. The full model adds exogenous streams, the available TRMF-inspired implementation, coupled loss, and physical projection. Each fused configuration selects its temporal weight on validation.

Every fold retains keyed predictions, per-model metrics, and a manifest with input/code hashes, partitions, seed, and model settings. `source_snapshot` retains the runner and model implementations used for the original keyed experiment. `environment.json` records the package versions. ETS metadata additionally identifies its tuning script. Results are single-seed point estimates, with no bootstrap significance claim.

Country diagnostics use only the aggregate accommodation category. Recovery cases are the median country within each tercile of the pre-test 2023/2019 recovery ratio. LOS errors are computed directly from predicted nights divided by predicted arrivals. Undefined predicted ratios are counted separately. The country LOS CSV includes all destination-level outcomes.

These are retrospective reference-period experiments. Source-stage covariate enrichment and missing historical publication/vintage records limit point-in-time interpretation. The new runner does not supply the missing release-vintage backtest.
