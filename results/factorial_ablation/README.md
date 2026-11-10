# Eurostat factorial ablation

The experiment fits eight combinations of three temporal switches with seeds 42,
43, and 44 on the frozen Eurostat CSV ending in July 2026.

| Switch | Off | On |
| --- | --- | --- |
| E | Direct exogenous projections disabled | Future calendar and past macro/COVID projections enabled together |
| H | Constant state ID 0 | HMM state filtered through the forecast origin |
| F | Additive state head only | Additive state head and multiplicative FiLM gain |

The additive state head and coupled loss are active in all eight cells. Plain
DLinear is an extra reference with both disabled. An enabled HMM can still
carry price and COVID information when the direct exogenous streams are off.

The fits use the benchmark's five-epoch protocol, frozen trend and seasonal
projections, AdamW learning rate 0.0001, batch size 256, 24-month lookback, and
12-month horizon. Each seed uses the all-on model to select its sigmoid fusion
weight on 2022 validation. The projected fused forecasts share this weight,
the same training-fitted TRMF forecast, and the same physical projection. Each
fit saves raw temporal predictions and projected fused predictions for the
same 4,522 observed test pairs.

Use a fresh output directory when fitting against a new data or code version.

```powershell
foreach ($seed in 42, 43, 44) {
    python scripts/factorial_ablation.py --seed $seed --output results/factorial_ablation_rerun
}
python scripts/factorial_ablation.py --output results/factorial_ablation_rerun --summarize
```

The summarizer checks that keys, origins, horizons, scoring masks, and observed
targets match. It recalculates WAPE and MAE and checks that the fusion weight
is common to the cells within each seed. Conditional contrasts switch one
flag while holding the other two fixed. The reported main effect averages the
four conditional contrasts within each seed before averaging across seeds.
Standard deviations describe seed variability on the recent retrospective fold.
