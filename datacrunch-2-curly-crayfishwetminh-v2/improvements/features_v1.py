"""
Feature engineering v1 for CrunchDAO DataCrunch #2.

Schema findings (from EDA on data/X.reduced.parquet):
  * 1,637,276 rows x 1,150 anonymized features (Feature_1 ... Feature_1150) + id + moon
  * 781 moons, ~2,096 rows per moon
  * Every `id` appears in exactly ONE moon (median=mean=max=1). `id` is a row
    identifier WITHIN a moon, not a security identifier persistent across time.
  * Each Feature_i takes only 7 unique values in [0, 1] -- features are
    already PER-MOON RANK-NORMALIZED into 7 quantile bins ({0, 1/6, 2/6, ..., 1}).
    Per-moon mean = 0.5 and per-moon std = 0.2319 for every feature.
  * No NaNs in features.

Implications for the 5 originally-requested techniques:
  1. Per-moon z-score          --> NEAR NO-OP. Raw features are already
                                   per-moon rank-normalized. z-score = (x - 0.5) / 0.232,
                                   a pure affine map that LightGBM is invariant to.
                                   Will produce ZERO score gain on tree models.
  2. Per-moon percentile rank  --> NEAR NO-OP. Raw features are already
                                   the rank in 7 buckets. We provide a true
                                   percentile (averages ties to break the 7-bin
                                   plateau) as a tiny refinement, but expect
                                   marginal gain only.
  3. Lag features (moon-1, -5) --> INFEASIBLE as specified. There is no
                                   id-to-id link across moons. We instead
                                   provide per-moon AGGREGATE lags (mean/std
                                   of each feature in moon-k joined back by
                                   moon), which captures regime context but
                                   is identical across all rows of a moon.
                                   In practice this only helps if combined with
                                   row-level features (e.g. as cross-features
                                   downstream). Off by default to keep the
                                   output a function of the row alone.
  4. Rolling mean/std per id    --> INFEASIBLE (each id appears once). Provided
                                   as no-op stub so the API matches the spec.
  5. Feature interactions 10x10 --> KEPT. With pre-binned features, pairwise
                                   products and absolute differences capture
                                   non-monotonic two-feature joint patterns
                                   that LightGBM splits cannot find in 200
                                   trees with colsample_bytree=0.5.
                                   This is the LARGEST EXPECTED GAIN.

Additional features added (NOT in the original list, but they exploit the
discretized structure that the original list does not):

  6. Row-wise SUMMARIES across all 1150 features (mean, std, skew, kurtosis,
     count of extreme bins). Because every feature is on the same [0,1]
     7-bin scale, these summaries are meaningful and cheap. They capture the
     row's "tilt" (how many features are in the top vs bottom bins).
  7. EXTREME-BIN COUNTS: per-row count of features at bin 0 (==0.0) and
     bin 1 (==1.0). Strong signal because extreme-bin assignment in a
     quantile-normalized regime usually indicates a security with consistent
     directional exposure.

ENTRY POINT:
    enhance_features(X, y_moon_col='moon', id_col='id',
                     top_features=None, include_interactions=True,
                     include_aggregates=True, include_row_summary=True) -> X_enhanced

`top_features` may be a list of feature column names to use for the 10x10
interaction block. If None, the first 10 features (Feature_1..Feature_10) are
used as a placeholder -- in production the caller should pass the top-10 by
LightGBM gain from a prior baseline fit.

The returned DataFrame contains:
    - all original columns (id, moon, Feature_1..Feature_1150)
    - new columns prefixed with `fe_`
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_cols(X: pd.DataFrame, id_col: str, moon_col: str) -> list[str]:
    return [c for c in X.columns if c not in (id_col, moon_col)
            and not c.startswith("fe_")]


# ---------------------------------------------------------------------------
# 1. Per-moon z-score (documented near no-op for tree models on pre-binned data)
# ---------------------------------------------------------------------------

def per_moon_zscore(X: pd.DataFrame, feature_cols: list[str],
                    moon_col: str) -> pd.DataFrame:
    """Returns a DataFrame of z-scored features with `fe_z_` prefix.

    For the actual DataCrunch features this is an affine transform of the
    input and adds ~0 lift on a GBM. Kept for completeness / for the case
    when the raw features change (i.e. removal of pre-binning upstream).
    """
    grp = X.groupby(moon_col, sort=False)[feature_cols]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)
    out = (X[feature_cols].astype(np.float32) - mu) / sd
    out = out.fillna(0.0).astype(np.float32)
    out.columns = [f"fe_z_{c}" for c in feature_cols]
    return out


# ---------------------------------------------------------------------------
# 2. Per-moon percentile rank (true average-rank, breaks the 7-bin plateau)
# ---------------------------------------------------------------------------

def per_moon_rank(X: pd.DataFrame, feature_cols: list[str],
                  moon_col: str) -> pd.DataFrame:
    """Per-moon percentile rank in [0, 1]. Average method breaks ties so the
    result has more than 7 unique values within each moon, slightly enriching
    the feature compared to the raw bin id."""
    grp = X.groupby(moon_col, sort=False)[feature_cols]
    out = grp.rank(method="average", pct=True).astype(np.float32)
    out.columns = [f"fe_rk_{c}" for c in feature_cols]
    return out


# ---------------------------------------------------------------------------
# 3. Moon-aggregate "lag" features (per-moon mean/std at moon-k, joined back).
# ---------------------------------------------------------------------------

def per_moon_aggregate_lag(X: pd.DataFrame, feature_cols: list[str],
                           moon_col: str, lags: Iterable[int] = (1, 5)
                           ) -> pd.DataFrame:
    """For each (feature, lag), compute the moon-level mean and std, then
    shift by `lag` moons and broadcast back to every row of the current moon.

    NOTE: row-level lag is INFEASIBLE because ids do not persist across
    moons. This function gives a regime-level proxy. The output is constant
    within a moon; it only adds value when combined with row-level features
    (LightGBM can use it as a context split).
    """
    out = {}
    moon_stats = X.groupby(moon_col)[feature_cols].agg(["mean", "std"])
    # moon_stats: index=moon, columns=MultiIndex(feature, stat)
    for lag in lags:
        shifted = moon_stats.shift(lag)
        for c in feature_cols:
            mu_key = f"fe_lag{lag}_mu_{c}"
            sd_key = f"fe_lag{lag}_sd_{c}"
            out[mu_key] = X[moon_col].map(shifted[(c, "mean")]).astype(np.float32)
            out[sd_key] = X[moon_col].map(shifted[(c, "std")]).astype(np.float32)
    df = pd.DataFrame(out, index=X.index)
    df = df.fillna(0.0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# 4. Per-id rolling stats -- INFEASIBLE here, stub kept for API parity.
# ---------------------------------------------------------------------------

def per_id_rolling(X: pd.DataFrame, feature_cols: list[str],
                   id_col: str, moon_col: str,
                   window: int = 5) -> pd.DataFrame:
    """Stub. In this dataset every `id` appears in exactly one moon, so
    per-id rolling stats are not defined. Returns an empty-column frame
    (zero-width) so the caller can concat it unconditionally."""
    return pd.DataFrame(index=X.index)


# ---------------------------------------------------------------------------
# 5. Pairwise interactions across top-K features (10x10 = 90 unique pairs).
# ---------------------------------------------------------------------------

def pairwise_interactions(X: pd.DataFrame, top_features: list[str]
                          ) -> pd.DataFrame:
    """For each unordered pair (a, b) in `top_features` (a != b), produce:
        fe_x_{a}_{b}   = a * b
        fe_d_{a}_{b}   = |a - b|
    Both capture conjunction / divergence patterns that single splits
    cannot express directly.
    """
    out = {}
    arr = {c: X[c].astype(np.float32).values for c in top_features}
    n = len(top_features)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = top_features[i], top_features[j]
            out[f"fe_x_{a}_{b}"] = arr[a] * arr[b]
            out[f"fe_d_{a}_{b}"] = np.abs(arr[a] - arr[b])
    return pd.DataFrame(out, index=X.index).astype(np.float32)


# ---------------------------------------------------------------------------
# 6+7. Row-wise summaries across all features (mean/std/skew/extreme-bin counts).
# ---------------------------------------------------------------------------

def row_summaries(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-row aggregates over all 1150 features. Because every feature is
    on the same [0,1] 7-bin scale, these aggregates have a consistent
    interpretation (cross-sectional "tilt" of the row).
    """
    arr = X[feature_cols].astype(np.float32).values
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    # Higher moments
    centered = arr - mean[:, None]
    var = (centered ** 2).mean(axis=1) + 1e-12
    skew = (centered ** 3).mean(axis=1) / (var ** 1.5)
    kurt = (centered ** 4).mean(axis=1) / (var ** 2) - 3.0
    # Extreme-bin counts: with 7 bins {0, 1/6, ..., 1}, the extremes are 0 and 1
    tol = 1e-4
    n_min = (arr <= tol).sum(axis=1).astype(np.float32)
    n_max = (arr >= 1.0 - tol).sum(axis=1).astype(np.float32)
    n_mid = ((arr > 0.45) & (arr < 0.55)).sum(axis=1).astype(np.float32)
    out = pd.DataFrame({
        "fe_row_mean": mean,
        "fe_row_std": std,
        "fe_row_skew": skew.astype(np.float32),
        "fe_row_kurt": kurt.astype(np.float32),
        "fe_row_nmin": n_min,
        "fe_row_nmax": n_max,
        "fe_row_nmid": n_mid,
        "fe_row_tilt": (n_max - n_min) / float(arr.shape[1]),  # net top vs bottom exposure
    }, index=X.index)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def enhance_features(
    X: pd.DataFrame,
    y_moon_col: str = "moon",
    id_col: str = "id",
    top_features: Optional[list[str]] = None,
    include_zscore: bool = False,       # default OFF: documented no-op for trees
    include_rank: bool = False,         # default OFF: marginal gain on pre-binned data
    include_lag: bool = False,          # default OFF: row-constant within moon, optional
    include_interactions: bool = True,  # ON: largest expected gain
    include_row_summary: bool = True,   # ON: cheap and complementary
) -> pd.DataFrame:
    """Append engineered features to X.

    Parameters
    ----------
    X : DataFrame with id, moon, Feature_1..Feature_N columns.
    y_moon_col, id_col : column names.
    top_features : list of feature names to use for the 10x10 interaction
        block. If None, Feature_1..Feature_10 are used (placeholder).
    include_* : toggles for each block.

    Returns
    -------
    DataFrame with the same row order as X, original columns preserved, and
    new feature columns prefixed `fe_`.
    """
    moon_col = y_moon_col
    feature_cols = _feature_cols(X, id_col, moon_col)
    if top_features is None:
        top_features = feature_cols[:10]
    else:
        # Validate; silently keep only the ones present.
        top_features = [c for c in top_features if c in feature_cols][:10]
        if len(top_features) < 2:
            top_features = feature_cols[:10]

    parts: list[pd.DataFrame] = [X]
    if include_zscore:
        parts.append(per_moon_zscore(X, feature_cols, moon_col))
    if include_rank:
        parts.append(per_moon_rank(X, feature_cols, moon_col))
    if include_lag:
        parts.append(per_moon_aggregate_lag(X, feature_cols, moon_col,
                                            lags=(1, 5)))
    # Per-id rolling intentionally always returns empty -- skip.
    if include_interactions:
        parts.append(pairwise_interactions(X, top_features))
    if include_row_summary:
        parts.append(row_summaries(X, feature_cols))

    out = pd.concat(parts, axis=1)
    return out


# ---------------------------------------------------------------------------
# Tests (run with `python improvements/features_v1.py`)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Sanity-check on a 1000-row sample of the real data."""
    import os
    import time

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    x_path = os.path.join(here, "data", "X.reduced.parquet")
    y_path = os.path.join(here, "data", "y.reduced.parquet")
    print(f"[test] reading {x_path}")
    # Grab a small contiguous sample spanning a few moons so that per-moon
    # groupby operations have multiple groups.
    X_full = pd.read_parquet(x_path)
    # Take ~1000 rows across 5 consecutive moons:
    moons_to_use = X_full["moon"].drop_duplicates().sort_values().iloc[10:15].tolist()
    X = X_full[X_full["moon"].isin(moons_to_use)].head(1000).reset_index(drop=True)
    print(f"[test] sample X shape={X.shape}, moons={sorted(X['moon'].unique())}")

    # Pretend top features are Feature_1..Feature_10
    top10 = [f"Feature_{i}" for i in range(1, 11)]

    t0 = time.time()
    Xe = enhance_features(X, top_features=top10,
                          include_zscore=True, include_rank=True,
                          include_lag=True, include_interactions=True,
                          include_row_summary=True)
    dt = time.time() - t0
    print(f"[test] enhance_features (all blocks ON) took {dt:.2f}s")
    print(f"[test] output shape: {Xe.shape}  (added {Xe.shape[1] - X.shape[1]} cols)")

    fe_cols = [c for c in Xe.columns if c.startswith("fe_")]
    by_prefix = {}
    for c in fe_cols:
        key = c.split("_")[1] if c.startswith("fe_") else "?"
        by_prefix[key] = by_prefix.get(key, 0) + 1
    print(f"[test] fe_ columns by group: {by_prefix}")

    # Schema integrity
    assert len(Xe) == len(X), "row count changed"
    assert Xe[["id", "moon"]].equals(X[["id", "moon"]]), "id/moon order changed"
    assert not Xe.isna().any().any(), "NaNs in output"
    print("[test] schema integrity OK (no NaNs, id/moon preserved)")

    # Sanity: row_mean equals manual mean over the 1150 features
    feat = [c for c in X.columns if c.startswith("Feature_")]
    manual_mean = X[feat].astype(np.float32).mean(axis=1).values
    assert np.allclose(manual_mean, Xe["fe_row_mean"].values, atol=1e-5), \
        "fe_row_mean mismatch"
    print("[test] fe_row_mean matches manual computation")

    # Sanity: 10x10 interactions -> 10*9/2 * 2 = 90 columns
    inter_cols = [c for c in Xe.columns if c.startswith("fe_x_") or c.startswith("fe_d_")]
    assert len(inter_cols) == 90, f"expected 90 interaction cols, got {len(inter_cols)}"
    print(f"[test] interaction count OK ({len(inter_cols)} cols)")

    # Sanity: z-score has near-zero mean per moon (within numerical noise)
    z0 = Xe.groupby("moon")["fe_z_Feature_1"].mean().abs().max()
    print(f"[test] |mean| of fe_z_Feature_1 within each moon (max across moons): {z0:.2e}")
    assert z0 < 1e-3, "z-score should have ~0 per-moon mean"

    # Default-toggle run (interactions + summaries only)
    t0 = time.time()
    Xe2 = enhance_features(X, top_features=top10)
    dt = time.time() - t0
    print(f"[test] enhance_features (defaults) took {dt:.2f}s -> shape {Xe2.shape}")

    print("[test] ALL TESTS PASSED")


if __name__ == "__main__":
    _self_test()
