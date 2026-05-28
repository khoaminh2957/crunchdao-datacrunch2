"""Target engineering strategies for CrunchDAO DataCrunch #2.

Scoring = per-moon Spearman correlation. Target is bounded [-1, 1], ~88% zeros,
already per-moon zero-mean (mean ~= 0 within each moon, std ~= 0.28).

Four transform strategies (each: y_df -> y_transformed_df, preserves [id, moon]).
Inverse provided for #1 and #2 (rank-invariant transforms don't need inverse for
Spearman scoring, but useful for offline calibration).

Usage:
    from improvements.target_v1 import zscore_per_moon, rank_per_moon, tanh_squash, quantile_bins
    y_t = rank_per_moon(y_train)
    model.fit(X, y_t['target'])
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module=__name__)


# ---------------------------------------------------------------------------
# 1. Per-moon z-score
# ---------------------------------------------------------------------------
def zscore_per_moon(y_df: pd.DataFrame, moon_col: str = "moon", target_col: str = "target",
                    eps: float = 1e-9) -> pd.DataFrame:
    """Standardize target within each moon: (y - mean_moon) / std_moon.

    Note: target is already ~zero-mean per moon, so this mostly equals y/std_moon.
    Useful when moon-to-moon volatility varies (rescales all moons to comparable scale).
    """
    out = y_df.copy()
    grp = out.groupby(moon_col)[target_col]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    out[target_col] = (out[target_col] - mean) / (std + eps)
    out[target_col] = out[target_col].fillna(0.0).astype(np.float32)
    return out


def zscore_per_moon_inverse(y_t_df: pd.DataFrame, y_orig_df: pd.DataFrame,
                             moon_col: str = "moon", target_col: str = "target") -> pd.DataFrame:
    """Invert per-moon z-score using stats from the original training y."""
    stats = y_orig_df.groupby(moon_col)[target_col].agg(["mean", "std"]).reset_index()
    merged = y_t_df.merge(stats, on=moon_col, how="left", suffixes=("", "_s"))
    out = y_t_df.copy()
    out[target_col] = (merged[target_col].values * merged["std"].values) + merged["mean"].values
    return out


# ---------------------------------------------------------------------------
# 2. Per-moon rank -> [0, 1]
# ---------------------------------------------------------------------------
def rank_per_moon(y_df: pd.DataFrame, moon_col: str = "moon", target_col: str = "target",
                  method: str = "average") -> pd.DataFrame:
    """Convert target to percentile rank within each moon, in [0, 1].

    THIS IS THE Spearman-NATIVE TRANSFORM: training on ranks directly optimizes
    the metric (Spearman = Pearson on ranks). Heavily clipped/zero-inflated
    distributions become uniform.
    """
    out = y_df.copy()
    # rank then normalize by group size -> percentile in (0, 1]
    out[target_col] = (out.groupby(moon_col)[target_col]
                          .rank(method=method, na_option="keep") /
                       out.groupby(moon_col)[target_col].transform("count"))
    out[target_col] = out[target_col].astype(np.float32)
    return out


def rank_per_moon_inverse(y_t_df: pd.DataFrame, y_orig_df: pd.DataFrame,
                           moon_col: str = "moon", target_col: str = "target") -> pd.DataFrame:
    """Approximate inverse: map percentile back to the empirical quantile of the
    original per-moon distribution. Only meaningful when the moons in y_t_df
    overlap with y_orig_df."""
    out = y_t_df.copy()
    new_vals = np.empty(len(out), dtype=np.float32)
    for m, idx in out.groupby(moon_col).groups.items():
        if m in y_orig_df[moon_col].values:
            ref = y_orig_df.loc[y_orig_df[moon_col] == m, target_col].dropna().values
            new_vals[idx] = np.quantile(ref, out.loc[idx, target_col].clip(0, 1).values)
        else:
            new_vals[idx] = out.loc[idx, target_col].values
    out[target_col] = new_vals
    return out


# ---------------------------------------------------------------------------
# 3. Tanh squashing (clip extreme values)
# ---------------------------------------------------------------------------
def tanh_squash(y_df: pd.DataFrame, moon_col: str = "moon", target_col: str = "target",
                scale: float = 3.0) -> pd.DataFrame:
    """Apply tanh(y / scale) elementwise. Compresses tails toward +/- 1.

    Because the raw target is already in [-1, 1], we first standardize per moon
    (so 'scale' is in std units), then squash. scale=3 keeps the middle ~linear
    and crushes outliers > 3 sigma.
    """
    z = zscore_per_moon(y_df, moon_col=moon_col, target_col=target_col)
    z[target_col] = np.tanh(z[target_col].values / scale).astype(np.float32)
    return z


# ---------------------------------------------------------------------------
# 4. Quantile binning (10-bin discretization)
# ---------------------------------------------------------------------------
def quantile_bins(y_df: pd.DataFrame, moon_col: str = "moon", target_col: str = "target",
                  n_bins: int = 10) -> pd.DataFrame:
    """Discretize target into n_bins per-moon quantile buckets, labeled 0..n_bins-1.

    Useful for converting regression -> ordinal classification. For Spearman
    scoring, bin indices preserve rank order (ties within bin OK; Spearman uses
    average ranks). When the target is heavily zero-inflated (~88% zeros), most
    bins will collide on 0 -> effective n_bins << 10. Returns int8.
    """
    out = y_df.copy()

    def _bin(s):
        try:
            return pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.zeros(len(s), dtype=np.int64), index=s.index)

    out[target_col] = (out.groupby(moon_col)[target_col]
                          .transform(_bin)
                          .fillna(-1)
                          .astype(np.int8))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _report(name, original, transformed, target_col="target"):
    o = original[target_col].astype(float)
    t = transformed[target_col].astype(float)
    print(f"  {name:22s} | mean {o.mean():+.4f} -> {t.mean():+.4f} | "
          f"std {o.std():.4f} -> {t.std():.4f} | "
          f"min {o.min():+.3f} -> {t.min():+.3f} | "
          f"max {o.max():+.3f} -> {t.max():+.3f} | "
          f"n_unique {o.nunique()} -> {t.nunique()}")


def _self_test():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    y_path = os.path.join(here, "data", "y.reduced.parquet")
    print(f"[self-test] loading {y_path}")
    y = pd.read_parquet(y_path)
    # take 20 moons for speed
    sample_moons = sorted(y["moon"].unique())[:20]
    y_s = y[y["moon"].isin(sample_moons)].copy()
    print(f"[self-test] sample: {len(y_s)} rows across {len(sample_moons)} moons")
    print(f"[self-test] pct_zero in target: {(y_s['target']==0).mean():.3f}")
    print()
    print("transform                | mean change                 | std change          | "
          "min/max change                       | n_unique change")
    print("-" * 130)
    _report("1. zscore_per_moon",   y_s, zscore_per_moon(y_s))
    _report("2. rank_per_moon",     y_s, rank_per_moon(y_s))
    _report("3. tanh_squash(s=3)",  y_s, tanh_squash(y_s, scale=3.0))
    _report("4. quantile_bins(10)", y_s, quantile_bins(y_s, n_bins=10))
    print()

    # round-trip check for inverses
    print("[self-test] inverse round-trip:")
    z = zscore_per_moon(y_s)
    z_back = zscore_per_moon_inverse(z, y_s)
    diff_z = (z_back["target"].astype(float) - y_s["target"].astype(float)).abs().max()
    print(f"  zscore round-trip max |err|: {diff_z:.2e}")

    r = rank_per_moon(y_s)
    r_back = rank_per_moon_inverse(r, y_s)
    # rank inverse is approximate -> compare Spearman correlation per moon
    from scipy.stats import spearmanr
    corrs = []
    for m in sample_moons[:5]:
        a = y_s.loc[y_s["moon"] == m, "target"].values
        b = r_back.loc[r_back["moon"] == m, "target"].values
        if len(a) > 1:
            c, _ = spearmanr(a, b)
            corrs.append(c)
    print(f"  rank inverse Spearman vs original (first 5 moons): "
          f"{[f'{c:.4f}' for c in corrs]}")


if __name__ == "__main__":
    _self_test()
