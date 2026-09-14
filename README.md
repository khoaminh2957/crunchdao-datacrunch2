# DataCrunch #2: Cross-Sectional Stock Return Prediction

My entry for CrunchDAO's DataCrunch #2 tournament (May 2026).

## Task

- **Goal:** predict next-period returns for about 1,900 stocks in each time period ("moon").
- **Inputs:** 1,150 anonymized features, quantile-binned per moon, over 790+ moons.
- **Scoring:** submissions run on CrunchDAO's cloud and are scored by the Pearson correlation between predictions and realized returns on a held-out moon.

## Approach

- **Model:** XGBoost on the raw target (500 trees, depth 6, learning rate 0.03), plus seed ensembles and a heterogeneous XGBoost + LightGBM + Ridge ensemble.
- **Research:** exploratory analysis of the feature structure (quantile binning, industry-like feature clusters), hyperparameter sweeps on GPU, and alternatives such as rank targets, 3-class classification, recency-weighted training, and MLPs.

## Validation

My first local validator (Spearman on rank targets, 1-moon gap) was badly miscalibrated: a deep XGBoost scored 0.082 locally but only 0.018 live.

I rebuilt it to match the live scoring more closely:
- Pearson correlation on the last test moon only
- 10 walk-forward folds with a 4-moon embargo
- raw target instead of ranks

Before each submission I recorded the predicted live score, then compared it with the actual score (`notes/prediction_ledger.md`).

## Results

Live scores on CrunchDAO's cloud:

| Submission | Model | Predicted | Live |
|---|---|---|---|
| v52 | XGBoost, depth 8 (old validator) | 0.082 | 0.018 |
| v60–v62, v66–v67 | XGBoost, single seeds | 0.065 | 0.023–0.081 (mean 0.060) |
| v63 | XGBoost, 3-seed average | 0.065 | 0.062 |
| v64 | XGBoost, 5-seed ensemble | 0.065 | 0.064 |
| v65 | 9-model XGBoost ensemble | 0.065 | 0.061 |
| v68 | XGBoost + LightGBM + Ridge | 0.067 | 0.076 |

**What I learned**
- **Seed noise is large.** Single-seed scores ranged from 0.023 to 0.081. Seed ensembles reduced the spread to about ±0.002.
- **The heterogeneous ensemble gave a real gain** over single XGBoost models.
- **Several ideas did not help:** training only on recent moons, 3-class classification, per-moon rank features, and trees of depth 8 or more.

**Outcome:** my best live score was 0.081. The top of the leaderboard at the time was 0.110, so this approach did not reach the top ranks.

## Repository layout

```
submissions/          code submitted to CrunchDAO (train/infer entry points per project)
experiments/          EDA, GPU sweeps, validators, and ensemble experiments
experiments/variants/ model variants tested before submission
notes/                pre-registered prediction ledger with live results
```

Competition data is not included. It is downloaded from CrunchDAO, and the experiment scripts expect `X.reduced.parquet` and `y.reduced.parquet` locally.
