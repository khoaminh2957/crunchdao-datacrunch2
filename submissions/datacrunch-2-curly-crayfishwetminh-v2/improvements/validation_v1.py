"""Purged time-series CV for CrunchDAO DataCrunch #2.

Why embargo: targets are forward-looking returns spanning multiple moons. A row
at moon `t` carries information that leaks into moons [t+1 ... t+horizon]. If the
validation fold sits right next to the training fold, the model is effectively
peeking at labels from overlapping return windows -> inflated CV score, ugly live.
Embargo drops the `k` moons on each side of every val fold so train labels can
no longer overlap val labels in time.

Walk-forward purged K-fold:
- Group rows by moon, split unique moons into K contiguous chunks (val folds).
- For each val fold, training set = all OTHER moons MINUS an embargo window of
  `embargo` moons before and after the val span.
- Per-moon Spearman is averaged (this is what CrunchDAO scores you on).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def purged_kfold_split(moons, n_splits: int = 5, embargo: int = 10):
    """Yield (train_moons, val_moons) tuples of moon IDs.

    Args:
        moons: iterable of moon IDs (rows of the dataset, or unique moons).
        n_splits: number of CV folds.
        embargo: number of moons to drop on EACH side of the val fold from
                 the train set (purge + embargo combined).
    """
    unique_moons = np.sort(np.unique(np.asarray(moons)))
    n_moons = len(unique_moons)
    if n_moons < n_splits:
        raise ValueError(f"need >= {n_splits} unique moons, got {n_moons}")

    # Contiguous, roughly equal val chunks (walk-forward style).
    fold_edges = np.array_split(unique_moons, n_splits)
    for val_chunk in fold_edges:
        val_lo, val_hi = val_chunk.min(), val_chunk.max()
        embargo_lo = val_lo - embargo
        embargo_hi = val_hi + embargo
        train_mask = (unique_moons < embargo_lo) | (unique_moons > embargo_hi)
        train_moons = unique_moons[train_mask]
        yield train_moons, val_chunk


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    moon_col: str = "moon",
    target_col: str = "target",
    n_splits: int = 5,
    embargo: int = 10,
    feature_cols=None,
    verbose: bool = True,
):
    """Run purged K-fold CV and report per-moon Spearman aggregates.

    Returns dict with mean_spearman, std_spearman, hit_rate, n_eras, per_moon (DataFrame).
    """
    # Align on (id, moon).
    join_keys = [c for c in ("id", moon_col) if c in X.columns and c in y.columns]
    merged = X.merge(y[[*join_keys, target_col]], on=join_keys, how="inner")

    if feature_cols is None:
        feature_cols = [
            c for c in X.columns if c not in ("id", "Id", moon_col, "Moon", target_col)
        ]

    moons_all = merged[moon_col].values
    per_moon_records = []

    for fold_idx, (train_moons, val_moons) in enumerate(
        purged_kfold_split(moons_all, n_splits=n_splits, embargo=embargo), start=1
    ):
        train_mask = np.isin(moons_all, train_moons) & merged[target_col].notna().values
        val_mask = np.isin(moons_all, val_moons) & merged[target_col].notna().values
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            if verbose:
                print(f"[fold {fold_idx}] skipped (empty after embargo)")
            continue

        Xtr = merged.loc[train_mask, feature_cols]
        ytr = merged.loc[train_mask, target_col]
        Xva = merged.loc[val_mask, feature_cols]
        yva = merged.loc[val_mask, target_col]
        m_va = merged.loc[val_mask, moon_col]

        model.fit(Xtr, ytr)
        preds = model.predict(Xva)

        fold_df = pd.DataFrame({moon_col: m_va.values, "y": yva.values, "p": preds})
        for mn, grp in fold_df.groupby(moon_col):
            if grp["y"].nunique() < 2 or grp["p"].nunique() < 2:
                continue
            rho, _ = spearmanr(grp["p"], grp["y"])
            if np.isnan(rho):
                continue
            per_moon_records.append({"fold": fold_idx, "moon": int(mn), "spearman": rho})

        if verbose:
            fold_rhos = [r["spearman"] for r in per_moon_records if r["fold"] == fold_idx]
            tag = (
                f"train_moons={len(train_moons)} val_moons={len(val_moons)} "
                f"train_rows={train_mask.sum():,} val_rows={val_mask.sum():,}"
            )
            if fold_rhos:
                print(
                    f"[fold {fold_idx}] {tag} mean_rho={np.mean(fold_rhos):+.4f} "
                    f"n_eras={len(fold_rhos)}"
                )
            else:
                print(f"[fold {fold_idx}] {tag} (no scoreable moons)")

    per_moon = pd.DataFrame(per_moon_records)
    if per_moon.empty:
        result = dict(mean_spearman=np.nan, std_spearman=np.nan, hit_rate=np.nan, n_eras=0, per_moon=per_moon)
    else:
        rhos = per_moon["spearman"].values
        result = dict(
            mean_spearman=float(rhos.mean()),
            std_spearman=float(rhos.std(ddof=1)) if len(rhos) > 1 else 0.0,
            hit_rate=float((rhos > 0).mean()),
            n_eras=int(len(rhos)),
            per_moon=per_moon,
        )

    if verbose:
        print("\n=== Purged K-Fold CV summary ===")
        print(f"  folds        : {n_splits}    embargo: {embargo} moons")
        print(f"  n_eras       : {result['n_eras']}")
        print(f"  mean Spearman: {result['mean_spearman']:+.4f}")
        print(f"  std  Spearman: {result['std_spearman']:.4f}")
        print(f"  hit rate     : {result['hit_rate']:.1%} (eras with rho>0)")
        if result["n_eras"] > 1:
            sharpe = result["mean_spearman"] / max(result["std_spearman"], 1e-9)
            print(f"  CV sharpe    : {sharpe:+.3f}")
        print("================================\n")

    return result


# ---------------------------------------------------------------------------
# Smoke test: small sample (~10k rows) to confirm the plumbing works.
# ---------------------------------------------------------------------------
def _smoke_test():
    import os
    import time
    from lightgbm import LGBMRegressor

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X = pd.read_parquet(os.path.join(here, "data", "X.reduced.parquet"))
    y = pd.read_parquet(os.path.join(here, "data", "y.reduced.parquet"))
    print(f"[smoke] full X={X.shape}, y={y.shape}")

    # Take ~10k rows but spread across many moons so CV folds have content.
    moons_sorted = np.sort(X["moon"].unique())
    sample_moons = moons_sorted[::max(1, len(moons_sorted) // 40)][:40]
    X_s = X[X["moon"].isin(sample_moons)].copy()
    if len(X_s) > 10_000:
        X_s = X_s.groupby("moon", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), 10_000 // len(sample_moons)), random_state=0)
        )
    y_s = y[y["id"].isin(X_s["id"])].copy()
    print(f"[smoke] sample X={X_s.shape}, y={y_s.shape}, moons={X_s['moon'].nunique()}")

    feat_cols = [c for c in X_s.columns if c.startswith("Feature_")]
    # Tiny model so the smoke test stays fast.
    model = LGBMRegressor(
        n_estimators=50, learning_rate=0.1, num_leaves=15,
        min_data_in_leaf=20, verbose=-1, n_jobs=-1,
    )
    t0 = time.time()
    res = evaluate_model(
        model, X_s, y_s, moon_col="moon", target_col="target",
        n_splits=5, embargo=2, feature_cols=feat_cols,
    )
    print(f"[smoke] done in {time.time()-t0:.1f}s")
    return res


if __name__ == "__main__":
    _smoke_test()
