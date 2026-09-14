"""Era-aware (moon-aware) features for CrunchDAO DataCrunch #2.

Data has temporal structure via "moon" column. Models should know temporal context.

Five blocks below; each function has a clear signature + a small-sample test in
`__main__`. Designed to be plug-and-play with main.py's train(): just call the
needed `add_*` helper on X_train and pass the corresponding `sample_weight=`
into LGBMRegressor.fit().
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Moon as a feature: raw + sin/cos cyclic encoding
# ---------------------------------------------------------------------------
def add_moon_cyclic(
    df: pd.DataFrame,
    moon_col: str = "moon",
    period: int = 12,
    prefix: str = "moon",
) -> pd.DataFrame:
    """Add raw moon + sin/cos of moon mod `period` for cyclic seasonality.

    `period=12` assumes ~monthly seasonal cycle; tune to match domain (e.g. 4
    for quarterly, 52 for weekly-of-year, etc.). Adds three columns:
      - {prefix}_raw   : raw moon value (also keeps original column intact)
      - {prefix}_sin   : sin(2*pi*moon/period)
      - {prefix}_cos   : cos(2*pi*moon/period)

    Returns a NEW DataFrame (does not mutate input).
    """
    out = df.copy()
    m = out[moon_col].astype(float).to_numpy()
    out.loc[:, f"{prefix}_raw"] = m
    out.loc[:, f"{prefix}_sin"] = np.sin(2.0 * np.pi * m / period)
    out.loc[:, f"{prefix}_cos"] = np.cos(2.0 * np.pi * m / period)
    return out


# ---------------------------------------------------------------------------
# 2. Moon since start: linear time-since-event delta
# ---------------------------------------------------------------------------
def add_moon_since_start(
    df: pd.DataFrame,
    moon_col: str = "moon",
    origin: int | None = None,
    out_col: str = "moon_since_start",
) -> pd.DataFrame:
    """Add delta from a fixed origin moon (defaults to df[moon_col].min()).

    Useful as a monotone "time index" feature so trees can learn long-horizon
    regime drift (level non-stationarity). Pass an explicit `origin` if you
    want consistent values across train/test (recommended: use train-min).
    """
    out = df.copy()
    if origin is None:
        origin = int(out[moon_col].min())
    out.loc[:, out_col] = out[moon_col].astype(int) - int(origin)
    return out


# ---------------------------------------------------------------------------
# 3. Per-id features: how many moons of history exist for this id
# ---------------------------------------------------------------------------
def add_id_history_features(
    df: pd.DataFrame,
    id_col: str = "id",
    moon_col: str = "moon",
    out_cols: tuple[str, str, str] = (
        "id_history_len",      # # of moons observed for this id up to & incl current
        "id_first_moon",       # earliest moon this id was seen
        "id_moons_since_first",  # current_moon - first_moon (id-tenure)
    ),
) -> pd.DataFrame:
    """Compute id-level temporal context.

    For each (id, moon) row:
      - id_history_len      : running count (1, 2, 3, ...) of rows for this id
                              sorted by moon (i.e. how many prior observations
                              the model "knows" about this id, inclusive).
      - id_first_moon       : first moon this id appears (constant per id).
      - id_moons_since_first: current_moon - id_first_moon (id "age" in moons).

    Computed without leakage (uses cumcount on sorted moons within id).
    """
    out = df.copy()
    out = out.sort_values([id_col, moon_col], kind="stable").reset_index(drop=True)
    grp = out.groupby(id_col, sort=False)
    c_len, c_first, c_since = out_cols
    out.loc[:, c_len] = grp.cumcount() + 1
    out.loc[:, c_first] = grp[moon_col].transform("min").astype(int)
    out.loc[:, c_since] = out[moon_col].astype(int) - out[c_first]
    return out


# ---------------------------------------------------------------------------
# 4. Recency weighting for the loss: exp(-(moon_max - moon) / tau)
# ---------------------------------------------------------------------------
def recency_weights(
    moons: pd.Series | np.ndarray,
    tau: float = 50.0,
    moon_max: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Exponential-decay sample weights favouring recent moons.

    w_i = exp(-(moon_max - moon_i) / tau)

    Args:
      moons     : per-row moon values (any iterable of numbers).
      tau       : half-life-ish constant. Larger tau = flatter weights.
                  Recommended grid: {20, 50, 100, ∞}.
      moon_max  : reference "now" moon. Defaults to moons.max() so the most
                  recent training row gets weight 1.0.
      normalize : if True, rescale so mean(w)=1 (keeps LightGBM's effective
                  sample size comparable to unweighted training).

    Returns 1-D np.ndarray of float weights (same length as `moons`).

    Memory note: tau=20 → row at moon_max-60 has weight exp(-3)=0.05;
                 tau=50 → same row has weight exp(-1.2)=0.30.
    """
    m = np.asarray(moons, dtype=float)
    if moon_max is None:
        moon_max = float(m.max())
    w = np.exp(-(moon_max - m) / float(tau))
    if normalize and w.mean() > 0:
        w = w / w.mean()
    return w


# ---------------------------------------------------------------------------
# 5. Era-decay schedule: combine recency w/ a per-era boost
# ---------------------------------------------------------------------------
def era_decay_weights(
    moons: pd.Series | np.ndarray,
    tau: float = 50.0,
    floor: float = 0.1,
    moon_max: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Recency weights with a `floor` so far-past eras are damped, not zeroed.

    w_i = max( exp(-(moon_max - moon_i) / tau), floor )

    Useful when you still want some signal from old eras (for stability) but
    weighted lower. Floor=0.0 reduces to pure `recency_weights`.

    Args:
      moons     : per-row moon values.
      tau       : exp-decay constant (same semantics as recency_weights).
      floor     : minimum weight before normalization (0..1). 0.1 = old eras
                  retain 10% of a "current" era's influence.
      moon_max  : reference moon (defaults to max).
      normalize : rescale so mean(w)=1.

    Returns 1-D np.ndarray of weights.
    """
    m = np.asarray(moons, dtype=float)
    if moon_max is None:
        moon_max = float(m.max())
    w = np.exp(-(moon_max - m) / float(tau))
    w = np.maximum(w, float(floor))
    if normalize and w.mean() > 0:
        w = w / w.mean()
    return w


# ---------------------------------------------------------------------------
# Convenience: one-shot pipeline
# ---------------------------------------------------------------------------
def add_all_era_features(
    df: pd.DataFrame,
    id_col: str = "id",
    moon_col: str = "moon",
    period: int = 12,
    origin: int | None = None,
) -> pd.DataFrame:
    """Apply (1)+(2)+(3) in one call. Leaves moon_col untouched."""
    out = add_moon_cyclic(df, moon_col=moon_col, period=period)
    out = add_moon_since_start(out, moon_col=moon_col, origin=origin)
    out = add_id_history_features(out, id_col=id_col, moon_col=moon_col)
    return out


# ---------------------------------------------------------------------------
# Small-sample tests (run: python improvements/era_features_v1.py)
# ---------------------------------------------------------------------------
def _make_sample() -> pd.DataFrame:
    # 3 ids x variable history; moons 1..10
    rows = []
    rng = np.random.default_rng(0)
    for id_, first_moon in [("A", 1), ("B", 3), ("C", 7)]:
        for m in range(first_moon, 11):
            rows.append({"id": id_, "moon": m, "feat1": rng.normal()})
    return pd.DataFrame(rows)


def _test_moon_cyclic():
    df = _make_sample()
    out = add_moon_cyclic(df, period=12)
    assert {"moon_raw", "moon_sin", "moon_cos"}.issubset(out.columns)
    # sin^2 + cos^2 == 1
    s2c2 = out["moon_sin"] ** 2 + out["moon_cos"] ** 2
    assert np.allclose(s2c2, 1.0), s2c2.describe()
    # moon=12 cycles back to moon=0
    sin12 = np.sin(2 * np.pi * 12 / 12)
    assert abs(sin12) < 1e-9
    print("[ok] add_moon_cyclic")


def _test_moon_since_start():
    df = _make_sample()
    out = add_moon_since_start(df, origin=1)
    assert out["moon_since_start"].min() == 0
    assert out["moon_since_start"].max() == 9
    print("[ok] add_moon_since_start")


def _test_id_history():
    df = _make_sample()
    out = add_id_history_features(df)
    # id C first seen at moon 7
    c_rows = out[out["id"] == "C"].sort_values("moon")
    assert c_rows["id_first_moon"].iloc[0] == 7
    assert list(c_rows["id_history_len"]) == [1, 2, 3, 4]
    assert list(c_rows["id_moons_since_first"]) == [0, 1, 2, 3]
    # id A history length on its last row should be 10
    assert out[(out["id"] == "A") & (out["moon"] == 10)]["id_history_len"].iloc[0] == 10
    print("[ok] add_id_history_features")


def _test_recency_weights():
    moons = np.array([1, 50, 100], dtype=float)
    w20 = recency_weights(moons, tau=20.0, normalize=False)
    w50 = recency_weights(moons, tau=50.0, normalize=False)
    # most recent moon (100) has weight 1.0
    assert abs(w20[-1] - 1.0) < 1e-9
    assert abs(w50[-1] - 1.0) < 1e-9
    # tau=20 decays faster than tau=50 for the oldest row
    assert w20[0] < w50[0]
    # normalized version has mean ~ 1
    wn = recency_weights(moons, tau=50.0, normalize=True)
    assert abs(wn.mean() - 1.0) < 1e-9
    print(f"[ok] recency_weights  (tau20 oldest={w20[0]:.4f}, tau50 oldest={w50[0]:.4f})")


def _test_era_decay_weights():
    moons = np.arange(1, 101, dtype=float)
    w = era_decay_weights(moons, tau=20.0, floor=0.1, normalize=False)
    # Floor enforced
    assert w.min() >= 0.1 - 1e-12
    # Most recent ~ 1.0 pre-normalize
    assert abs(w[-1] - 1.0) < 1e-9
    # Old rows clamped to floor
    assert abs(w[0] - 0.1) < 1e-9
    print(f"[ok] era_decay_weights  (oldest={w[0]:.3f}, newest={w[-1]:.3f})")


def _test_pipeline():
    df = _make_sample()
    out = add_all_era_features(df)
    expected = {
        "moon_raw", "moon_sin", "moon_cos",
        "moon_since_start",
        "id_history_len", "id_first_moon", "id_moons_since_first",
    }
    missing = expected - set(out.columns)
    assert not missing, f"missing: {missing}"
    assert len(out) == len(df)
    print(f"[ok] add_all_era_features  (shape={out.shape}, new_cols={sorted(expected)})")


if __name__ == "__main__":
    _test_moon_cyclic()
    _test_moon_since_start()
    _test_id_history()
    _test_recency_weights()
    _test_era_decay_weights()
    _test_pipeline()
    print("\nALL TESTS PASSED")
