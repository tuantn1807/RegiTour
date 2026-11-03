# Eurostat: old versus new implementation

Old commit: `9df92d85e1bd6e26ddb73b5c8079ae7a0e0f46fa`. New: working tree identified by source hashes in metadata.json.

Both runs: train through 2022; validation through 2023; 4,864 identical test rows. L=24, H=12, epochs=35, seed=42.

| Version | Stage | Arrivals WAPE (%) | Nights WAPE (%) | Arrivals MAE | Nights MAE | Temporal weight |
|---|---|---:|---:|---:|---:|---:|
| old | temporal | 18.09 | 23.29 | 269,978.90 | 949,636.14 | -- |
| old | framework | 22.23 | 30.29 | 331,814.00 | 1,235,294.93 | 0.7000 |
| new | temporal | 82.28 | 80.27 | 1,227,930.63 | 3,273,069.59 | -- |
| new | framework | 36.57 | 47.00 | 545,863.35 | 1,916,579.17 | 0.0000 |

These are point estimates from single seeded runs; no bootstrap uncertainty or significance test was computed.
The comparison measures the complete implementation change, including each version's calibration, projection and fusion search. It does not isolate the causal effect of one component.
The existing preprocessing and HMM protocol is reused; this comparison alone does not validate that protocol.
Historical manuscript values (14.59% / 25.00%, stated n=4,712) are a separate reference and are not substituted for the old version's measured results above.

## Interpretation

Both networks contain 4,752 learnable parameters. The deterioration is not explained by a larger parameter count.

The old framework selects a temporal weight of 0.70; the new framework selects 0.00, so its final prediction is supplied entirely by TRMF in this run. The new temporal core itself has WAPE 82.28% / 80.27%, compared with 18.09% / 23.29% for the old temporal path (including its extra baseline blend and hard corrections).

The full framework's arrivals WAPE rises by 14.34 percentage points (22.23% to 36.57%); overnight-stays WAPE rises by 16.71 percentage points (30.29% to 47.00%). These runs show a regression in predictive accuracy. Matching the paper equation alone has not preserved the earlier implementation's performance; identifying the responsible changes requires validation-based component ablations.

The common metric filter retains 4,523 arrivals observations and 4,666 overnight-stay observations from the same 4,864 test records. The manuscript's historical 4,712-record result is not this paired experiment. There are no new bootstrap standard deviations.
