"""
Local scorer approximating CrunchDAO DataCrunch #2 scoring.

Pipeline per moon:
  1. Gaussianise predictions: Phi^-1((rank - 0.5) / N)
  2. Mean-zero (subtract per-moon mean)
  3. L1 normalize (divide by sum(|.|))
  4. Dot product with raw target -> per-moon alpha
  5. Mean across moons = primary score
  6. Optional: cumprod(per_moon + 1) - 1 = compounded return
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _gaussianise(x: np.ndarray) -> np.ndarray:
    """Rank-Gaussianise a 1-D array. Returns N(0,1)-like values."""
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=float)
    r = rankdata(x, method="average")
    u = (r - 0.5) / n
    return norm.ppf(u)


def _per_moon_alpha(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute alpha for a single moon."""
    if len(pred) < 2:
        return 0.0
    g = _gaussianise(pred)
    g = g - g.mean()                       # mean-zero
    denom = np.sum(np.abs(g))
    if denom <= 0:
        return 0.0
    w = g / denom                          # L1 normalize -> portfolio weights
    return float(np.dot(w, target))        # dot product with raw target


def score(
    predictions,
    target,
    moons=None,
    per_moon: bool = True,
    compounded: bool = False,
):
    """
    Parameters
    ----------
    predictions, target : array-like or pd.Series (aligned)
    moons               : array-like of moon ids (same length). If None, treat all rows as one moon.
    per_moon            : if True, returns dict with per-moon alphas + primary score.
    compounded          : if True, also report cumprod(alpha + 1) - 1.

    Returns
    -------
    dict with keys: 'score', 'per_moon' (Series), 'compounded' (optional float)
    """
    pred = np.asarray(predictions, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if moons is None:
        moons = np.zeros(len(pred), dtype=int)
    moons = np.asarray(moons)

    df = pd.DataFrame({"pred": pred, "target": tgt, "moon": moons})

    per_moon_alpha = (
        df.groupby("moon", sort=True)[["pred", "target"]]
          .apply(lambda g: _per_moon_alpha(g["pred"].values, g["target"].values))
    )

    out = {
        "score": float(per_moon_alpha.mean()),
        "per_moon": per_moon_alpha,
    }
    if compounded:
        out["compounded"] = float(np.prod(per_moon_alpha.values + 1.0) - 1.0)
    if not per_moon:
        return out["score"]
    return out


# ----------------------------- self-test ----------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N_MOONS = 20
    N_PER_MOON = 500
    rows = []
    for m in range(N_MOONS):
        t = rng.normal(0, 0.05, N_PER_MOON)        # synthetic targets
        rows.append(pd.DataFrame({"moon": m, "target": t}))
    df = pd.concat(rows, ignore_index=True)
    target = df["target"].values
    moons = df["moon"].values

    # Test 1: random preds
    rand_pred = rng.normal(0, 1, len(df))
    r1 = score(rand_pred, target, moons=moons, compounded=True)

    # Test 2: perfect preds (pred = target)
    r2 = score(target, target, moons=moons, compounded=True)

    # Test 3: inverted preds
    r3 = score(-target, target, moons=moons, compounded=True)

    # Test 4: noisy-but-signal preds (target + noise)
    noisy = target + rng.normal(0, 0.05, len(df))
    r4 = score(noisy, target, moons=moons, compounded=True)

    print(f"{'Test':<30} {'mean alpha':>14} {'compounded':>14}")
    print("-" * 60)
    print(f"{'Random preds':<30} {r1['score']:>14.6f} {r1['compounded']:>14.6f}")
    print(f"{'Perfect preds (pred=tgt)':<30} {r2['score']:>14.6f} {r2['compounded']:>14.6f}")
    print(f"{'Inverted preds (pred=-tgt)':<30} {r3['score']:>14.6f} {r3['compounded']:>14.6f}")
    print(f"{'Noisy preds (tgt+noise)':<30} {r4['score']:>14.6f} {r4['compounded']:>14.6f}")

    assert abs(r1["score"]) < 5e-3, "Random preds should score ~0"
    assert r2["score"] > 0, "Perfect preds should be positive"
    assert r3["score"] < 0, "Inverted preds should be negative"
    assert abs(r2["score"] + r3["score"]) < 1e-9, "Perfect and inverted must be exact opposites"
    print("\nAll assertions passed.")
