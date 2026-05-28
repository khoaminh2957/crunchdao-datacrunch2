"""
post_process.py — standardize predictions so cloud's preprocessing is a no-op.

Cloud applies (per moon, in order):
  1. Gaussianisation (rank -> norm.ppf)
  2. Mean-zero centering
  3. L1 normalization

If we apply the same transforms locally BEFORE returning, the cloud's
preprocessing becomes idempotent (no information loss / no surprises).

The ONLY information that survives this pipeline is per-moon rank order,
so the local model must produce correctly-ordered predictions.

Usage inside infer():
    pred_df = post_process(pred_df, moon_col='moon', pred_col='prediction')
    return pred_df
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def _gaussianise(x: np.ndarray) -> np.ndarray:
    """Map values -> standard-normal quantiles via rank. Ties get average rank."""
    n = len(x)
    if n == 0:
        return x
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    # pandas rank handles ties with method='average'
    ranks = pd.Series(x).rank(method='average').to_numpy()
    # Map ranks (1..n) -> uniform (0,1) using (r - 0.5) / n to avoid 0 and 1
    u = (ranks - 0.5) / n
    return norm.ppf(u)


def _mean_zero(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def _l1_normalize(x: np.ndarray) -> np.ndarray:
    s = np.abs(x).sum()
    if s == 0 or not np.isfinite(s):
        return x
    return x / s


def post_process(pred_df: pd.DataFrame,
                 moon_col: str = 'moon',
                 pred_col: str = 'prediction') -> pd.DataFrame:
    """Apply per-moon Gaussianise -> mean-zero -> L1 normalize on pred_col."""
    out = pred_df.copy()

    def _transform(group_vals: pd.Series) -> pd.Series:
        x = group_vals.to_numpy(dtype=np.float64)
        # Replace NaN/inf with median rank (treat as neutral) to keep pipeline stable
        if not np.all(np.isfinite(x)):
            finite = x[np.isfinite(x)]
            fill = np.median(finite) if finite.size else 0.0
            x = np.where(np.isfinite(x), x, fill)
        x = _gaussianise(x)
        x = _mean_zero(x)
        x = _l1_normalize(x)
        return pd.Series(x, index=group_vals.index)

    out[pred_col] = out.groupby(moon_col, sort=False, group_keys=False)[pred_col].apply(_transform)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # Build a synthetic frame: 3 moons x 100 stocks
    n_moons, n_stocks = 3, 100
    rows = []
    for m in range(n_moons):
        for s in range(n_stocks):
            rows.append({'moon': m, 'id': f'{m}_{s}', 'prediction': rng.normal()})
    df_random = pd.DataFrame(rows)

    # Test 1: random preds — post-processed must preserve rank order per moon
    pp_random = post_process(df_random.copy())
    rank_preserved_random = True
    for m, g in df_random.groupby('moon'):
        r_in = g['prediction'].rank(method='average').to_numpy()
        r_out = pp_random.loc[g.index, 'prediction'].rank(method='average').to_numpy()
        if not np.allclose(r_in, r_out):
            rank_preserved_random = False
            break

    # Test 2: sorted preds — post-processed must still be sorted within each moon
    df_sorted = df_random.copy()
    df_sorted['prediction'] = df_sorted.groupby('moon').cumcount().astype(float)
    pp_sorted = post_process(df_sorted.copy())
    sorted_preserved = True
    for m, g in pp_sorted.groupby('moon'):
        # Within each moon, sort by original cumcount (already the order in df_sorted)
        orig = df_sorted.loc[g.index, 'prediction'].to_numpy()
        new = g['prediction'].to_numpy()
        # Check: argsort(new) == argsort(orig)
        if not np.array_equal(np.argsort(orig), np.argsort(new)):
            sorted_preserved = False
            break

    # Test 3: per-moon stats — mean ~0, sum|x| ~1, ~gaussian-shaped
    stats = []
    for m, g in pp_random.groupby('moon'):
        x = g['prediction'].to_numpy()
        stats.append({
            'moon': m,
            'mean': float(x.mean()),
            'sum_abs': float(np.abs(x).sum()),
            'std_x_n': float(x.std() * len(x)),  # rough scale indicator
            'min': float(x.min()),
            'max': float(x.max()),
        })

    print('=' * 60)
    print('post_process.py self-test')
    print('=' * 60)
    print(f'Test 1 (random rank preserved):  {rank_preserved_random}')
    print(f'Test 2 (sorted order preserved): {sorted_preserved}')
    print('Test 3 (per-moon stats):')
    for s in stats:
        print(f"  moon={s['moon']}  mean={s['mean']:+.2e}  "
              f"sum|x|={s['sum_abs']:.6f}  min={s['min']:+.4f}  max={s['max']:+.4f}")
    print('=' * 60)

    assert rank_preserved_random, 'FAIL: rank order not preserved on random preds'
    assert sorted_preserved,      'FAIL: sorted order not preserved'
    for s in stats:
        assert abs(s['mean']) < 1e-10,           f"FAIL: mean!=0 on moon {s['moon']}"
        assert abs(s['sum_abs'] - 1.0) < 1e-10,  f"FAIL: L1!=1 on moon {s['moon']}"
    print('ALL TESTS PASSED')
