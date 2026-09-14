"""
Feature selection v1 — Agent 5 of 10.

Goal: reduce 1150 raw features to a tractable subset (100/200/300/500) for
downstream LightGBM training. Combines two ranking signals:

  (A) LightGBM gain importance fit on a small uniform-sample of rows.
  (B) Per-moon mean absolute Pearson correlation between feature and target
      (averaged across moons, weighted by per-moon row counts).

Outputs (written to improvements/):
  selected_features.json — dict with top-k lists for k in {100,200,300,500}
                            plus union/intersection helpers and metadata.
  feature_importance_v1.csv — full per-feature ranking table.

Run:
  python improvements/feature_select_v1.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "improvements"
X_PATH = DATA_DIR / "X.reduced.parquet"
Y_PATH = DATA_DIR / "y.reduced.parquet"

OUT_JSON = OUT_DIR / "selected_features.json"
OUT_CSV = OUT_DIR / "feature_importance_v1.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LGB_SAMPLE_ROWS = 50_000      # rows for LGBM importance fit
CORR_SAMPLE_MOONS = 120       # moons sampled for per-moon correlation
CORR_MAX_ROWS_PER_MOON = 4_000  # cap rows per moon (most moons are ~2k)
TOP_K_LIST = [100, 200, 300, 500]
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data():
    t0 = time.time()
    print(f"[load] reading {Y_PATH.name}")
    y = pd.read_parquet(Y_PATH)
    print(f"[load] y shape={y.shape}, cols={list(y.columns)}")

    print(f"[load] reading {X_PATH.name} (this is the slow step)")
    X = pd.read_parquet(X_PATH)
    print(f"[load] X shape={X.shape}, elapsed={time.time()-t0:.1f}s")

    # Merge on id+moon
    merged = X.merge(y, on=["id", "moon"], how="inner")
    print(f"[load] merged shape={merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# (A) LightGBM gain importance
# ---------------------------------------------------------------------------
def lgbm_importance(merged: pd.DataFrame, feature_cols: list[str], target_col: str):
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(RANDOM_SEED)
    mask = merged[target_col].notna()
    sub = merged.loc[mask]
    n = min(LGB_SAMPLE_ROWS, len(sub))
    idx = rng.choice(len(sub), size=n, replace=False)
    sample = sub.iloc[idx]
    print(f"[lgbm] sampled {len(sample)} rows for importance fit")

    Xt = sample[feature_cols]
    yt = sample[target_col]

    t0 = time.time()
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        importance_type="gain",
        verbose=-1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(Xt, yt)
    print(f"[lgbm] fit done in {time.time()-t0:.1f}s")

    imp = pd.Series(model.feature_importances_, index=feature_cols, name="lgbm_gain")
    return imp


# ---------------------------------------------------------------------------
# (B) Per-moon mean abs correlation
# ---------------------------------------------------------------------------
def per_moon_abs_corr(merged: pd.DataFrame, feature_cols: list[str], target_col: str):
    rng = np.random.default_rng(RANDOM_SEED + 1)
    all_moons = merged["moon"].unique()
    if len(all_moons) > CORR_SAMPLE_MOONS:
        moons = rng.choice(all_moons, size=CORR_SAMPLE_MOONS, replace=False)
    else:
        moons = all_moons
    print(f"[corr] using {len(moons)} moons (of {len(all_moons)} total)")

    acc = np.zeros(len(feature_cols), dtype=np.float64)
    weight = 0.0
    target_arr_cache = {}

    t0 = time.time()
    for i, m in enumerate(moons):
        chunk = merged[merged["moon"] == m]
        chunk = chunk[chunk[target_col].notna()]
        if len(chunk) < 50:
            continue
        if len(chunk) > CORR_MAX_ROWS_PER_MOON:
            chunk = chunk.sample(n=CORR_MAX_ROWS_PER_MOON, random_state=RANDOM_SEED)

        Xm = chunk[feature_cols].to_numpy(dtype=np.float64, copy=False)
        ym = chunk[target_col].to_numpy(dtype=np.float64, copy=False)

        # Standardize within moon, ignore zero-variance features
        Xm_mean = Xm.mean(axis=0)
        Xm_std = Xm.std(axis=0)
        Xm_std = np.where(Xm_std < 1e-12, 1.0, Xm_std)
        Xm_z = (Xm - Xm_mean) / Xm_std

        ym_mean = ym.mean()
        ym_std = ym.std()
        if ym_std < 1e-12:
            continue
        ym_z = (ym - ym_mean) / ym_std

        # Pearson correlation per feature
        corr = (Xm_z * ym_z[:, None]).mean(axis=0)
        acc += np.abs(corr) * len(chunk)
        weight += len(chunk)

        if (i + 1) % 20 == 0:
            print(f"[corr] processed {i+1}/{len(moons)} moons, "
                  f"elapsed={time.time()-t0:.1f}s")

    if weight == 0:
        raise RuntimeError("no usable moons for correlation")
    mean_abs_corr = acc / weight
    s = pd.Series(mean_abs_corr, index=feature_cols, name="per_moon_abs_corr")
    print(f"[corr] done in {time.time()-t0:.1f}s")
    return s


# ---------------------------------------------------------------------------
# Combine + write
# ---------------------------------------------------------------------------
def combine_and_save(lgbm_imp: pd.Series, corr_imp: pd.Series, feature_cols: list[str]):
    # Rank-normalize each signal to [0,1], then average for a combined score.
    lgbm_rank = lgbm_imp.rank(pct=True)
    corr_rank = corr_imp.rank(pct=True)
    combined = (lgbm_rank + corr_rank) / 2.0
    combined.name = "combined_rank"

    table = pd.concat([lgbm_imp, lgbm_rank.rename("lgbm_rank"),
                       corr_imp, corr_rank.rename("corr_rank"),
                       combined], axis=1)
    table.index.name = "feature"
    table = table.sort_values("combined_rank", ascending=False)
    table.to_csv(OUT_CSV)
    print(f"[save] wrote {OUT_CSV} ({len(table)} rows)")

    out = {
        "meta": {
            "n_features_input": len(feature_cols),
            "lgbm_sample_rows": LGB_SAMPLE_ROWS,
            "corr_sample_moons": CORR_SAMPLE_MOONS,
            "corr_max_rows_per_moon": CORR_MAX_ROWS_PER_MOON,
            "random_seed": RANDOM_SEED,
            "ranking_method": "mean of lgbm_gain pct-rank and per-moon abs-corr pct-rank",
        },
    }

    for k in TOP_K_LIST:
        top_combined = table.index[:k].tolist()
        top_lgbm = lgbm_imp.sort_values(ascending=False).index[:k].tolist()
        top_corr = corr_imp.sort_values(ascending=False).index[:k].tolist()
        out[f"top_{k}_combined"] = top_combined
        out[f"top_{k}_lgbm"] = top_lgbm
        out[f"top_{k}_corr"] = top_corr
        out[f"top_{k}_intersection"] = sorted(set(top_lgbm) & set(top_corr))

    # Convenience: default selection
    out["selected"] = out["top_300_combined"]
    out["selected_k"] = 300

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] wrote {OUT_JSON}")

    # Print top 20
    print("\n=== TOP 20 by combined rank ===")
    print(table.head(20).to_string(float_format=lambda x: f"{x:.4f}"))

    # Print overlap stats
    print("\n=== overlap stats ===")
    for k in TOP_K_LIST:
        a = set(out[f"top_{k}_lgbm"])
        b = set(out[f"top_{k}_corr"])
        inter = len(a & b)
        print(f"  k={k:>3}  lgbm∩corr = {inter:>3}  ({100*inter/k:.0f}%)")

    return table


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_data()

    feature_cols = [c for c in merged.columns
                    if c not in ("id", "moon", "target")]
    target_col = "target"
    print(f"[main] {len(feature_cols)} feature columns")

    lgbm_imp = lgbm_importance(merged, feature_cols, target_col)
    corr_imp = per_moon_abs_corr(merged, feature_cols, target_col)
    combine_and_save(lgbm_imp, corr_imp, feature_cols)
    print("\n[main] done.")


if __name__ == "__main__":
    main()
