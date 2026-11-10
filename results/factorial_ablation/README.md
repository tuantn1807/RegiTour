# Bias-controlled temporal ablation

This experiment supplements the original one-seed component ladder. It evaluates
all eight combinations of three switches on the Eurostat temporal branch:

| Switch | Off | On |
| --- | --- | --- |
| E | Exogenous projections disabled | Both future calendar and past macro/COVID streams enabled |
| H | Constant state ID 0 | Training-fitted, forward-filtered HMM state at the forecast origin |
| F | Additive state head only | Additive state head and multiplicative FiLM gain |

The additive head stays enabled in every factorial cell. Consequently, switching
H changes the state signal while retaining the same head architecture. The
E0H0F0 cell includes a learned constant intercept and is not plain DLinear. A
separate plain DLinear run disables the head, exogenous streams, and FiLM.
E removes only the direct projection streams. When H is enabled, its HMM can
still convey price and COVID information through the state signal. Thus these
are architectural channel contrasts, not effects of disjoint information sets.
F measures the multiplicative gain beyond the additive control, rather than
removing both parts of the original FiLM transformation at once.

All runs use seeds 42, 43, and 44, 25 epochs, batch size 256, AdamW learning rate
0.005, lookback 24, horizon 12, and moving-average kernel 13. These settings are
inherited from the reference experiment. There is no additional hyperparameter
search or model selection. Training ends in 2021. Validation in 2022 and residual
calibration in 2023 remain reserved and are not used in this raw point-forecast
experiment. Forecasts are issued at January origins in 2024, 2025, and 2026; the
last block ends in July 2026. All cells score the same 4,522 originally observed,
nonnegative target pairs from the 4,712-row grid.

The temporal weight is fixed at one. TRMF, coupled consistency loss, physical
projection, and residual intervals are excluded so they cannot conceal component
differences. These controls estimate temporal configuration effects under this
fixed training budget. They do not establish an incremental contribution to a
fused forecast that selects zero temporal weight.

To verify retained predictions and regenerate their tables, run this from the
repository root:

```powershell
python scripts/factorial_ablation.py --summarize
```

To repeat training, use a fresh output directory. Strict manifests prevent
resuming a seed with changed source hashes, including later rendering changes:

```powershell
python scripts/factorial_ablation.py --seed 42 --output results/factorial_ablation_reproduction
python scripts/factorial_ablation.py --seed 43 --output results/factorial_ablation_reproduction
python scripts/factorial_ablation.py --seed 44 --output results/factorial_ablation_reproduction
python scripts/factorial_ablation.py --output results/factorial_ablation_reproduction --summarize
```

Each seed folder retains keyed predictions, metrics, input/code hashes, origin
states, and the exact source files used. Summarization checks that every result
uses identical keys, origins, horizons, observed targets, and scoring masks, and
recomputes WAPE/MAE from predictions. Per-seed conditional contrasts switch one
flag while holding the other two fixed. Average effects first average the four
matched contrasts within each seed, then summarize the three seed estimates.
Reported standard deviations describe seed variability, not uncertainty over
future periods. No significance or causal mechanism claim follows from these
three seeds. Conditional effects can differ because of interactions and changes
in optimization or model capacity when enabling FiLM or exogenous projections.

Inputs remain latest-vintage, retrospective reference-period snapshots. This
experiment does not reconstruct historical publication delays or revisions and
does not complete the separate Original Availability versus PIT comparison.
