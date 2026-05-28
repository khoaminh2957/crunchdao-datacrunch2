"""
CrunchDAO DataCrunch #2 - LightGBM DART v1

Different boosting variant from baseline LGBM gbdt (score 0.0124).
DART adds dropout to tree boosting, which tends to reduce overfitting on
noisy regression at the cost of slower training. Same Spearman-native
rank-per-moon target and recency-weighted samples (tau=50, mean-normalized).

Public API:
    train(X_train, y_train, model_directory_path)
    infer(X_test, model_directory_path) -> pd.DataFrame[id, moon, prediction]
"""

from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


MODEL_FILENAME = "model.pkl"

LGBM_DART_PARAMS = dict(
    boosting_type="dart",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    drop_rate=0.1,
    max_drop=50,
    min_data_in_leaf=200,
    colsample_bytree=0.5,
    reg_lambda=2.0,
    verbose=-1,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return Feature_* columns in stable numeric-suffix order."""
    feats = [c for c in df.columns if c.startswith("Feature_")]
    feats.sort(key=lambda c: int(c.split("_", 1)[1]))
    return feats


def _rank_per_moon(y: pd.Series, moons: pd.Series) -> np.ndarray:
    """Rank target within each moon, scaled to [-1, 1] (Spearman-native)."""
    s = pd.Series(np.asarray(y, dtype=float), index=moons.index)
    ranked = s.groupby(moons).transform(
        lambda g: g.rank(method="average", pct=True)
    )
    return (ranked.to_numpy() * 2.0) - 1.0


def _recency_weights(moons: np.ndarray, tau: float = 50.0) -> np.ndarray:
    """exp(-(moon_max - moon)/tau), normalized so mean(w) = 1."""
    mmax = float(np.max(moons))
    w = np.exp(-(mmax - moons.astype(np.float64)) / tau)
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# train / infer
# ---------------------------------------------------------------------------
def train(X_train: pd.DataFrame, y_train, model_directory_path: str) -> None:
    """Fit LGBMRegressor (DART) on rank-per-moon target with recency weights."""
    os.makedirs(model_directory_path, exist_ok=True)

    if "moon" not in X_train.columns:
        raise ValueError("X_train must contain a 'moon' column for per-moon ranking.")

    if isinstance(y_train, pd.DataFrame):
        target_col = y_train.columns[0]
        y_series = y_train.iloc[:, 0]
    elif isinstance(y_train, pd.Series):
        target_col = y_train.name or "target"
        y_series = y_train
    else:
        target_col = "target"
        y_series = pd.Series(np.asarray(y_train), index=X_train.index)

    feature_cols = _feature_cols(X_train)
    if not feature_cols:
        raise ValueError("No Feature_* columns found in X_train.")

    y_ranked = _rank_per_moon(y_series, X_train["moon"])
    sample_w = _recency_weights(X_train["moon"].to_numpy(), tau=50.0)

    print(
        f"[train-dart] X={X_train.shape}, feats={len(feature_cols)}, "
        f"rank mean={y_ranked.mean():.3f} std={y_ranked.std():.3f}, "
        f"w mean={sample_w.mean():.3f} min={sample_w.min():.3f} max={sample_w.max():.3f}"
    )

    model = LGBMRegressor(**LGBM_DART_PARAMS)
    model.fit(
        X_train[feature_cols].to_numpy(dtype=np.float32),
        y_ranked,
        sample_weight=sample_w,
    )

    with open(os.path.join(model_directory_path, MODEL_FILENAME), "wb") as f:
        pickle.dump((model, feature_cols, target_col), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[train-dart] saved model + {len(feature_cols)} features (target='{target_col}')")


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    """Score X_test; returns DataFrame[id, moon, prediction] (scorer-required schema)."""
    with open(os.path.join(model_directory_path, MODEL_FILENAME), "rb") as f:
        model, feature_cols, _target_col = pickle.load(f)

    missing = [c for c in feature_cols if c not in X_test.columns]
    if missing:
        raise ValueError(f"X_test missing {len(missing)} expected features, e.g. {missing[:3]}")
    if "id" not in X_test.columns:
        raise ValueError("X_test must contain an 'id' column.")
    if "moon" not in X_test.columns:
        raise ValueError("X_test must contain a 'moon' column.")

    preds = model.predict(X_test[feature_cols].to_numpy(dtype=np.float32))

    out = pd.DataFrame(
        {
            "id": X_test["id"].to_numpy(),
            "moon": X_test["moon"].to_numpy(),
            "prediction": preds.astype(np.float64),
        }
    )
    return out[["id", "moon", "prediction"]]


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    n_rows = 1000
    n_feats = 1150
    n_moons = 20

    moons = rng.integers(0, n_moons, size=n_rows)
    feats = rng.standard_normal(size=(n_rows, n_feats)).astype(np.float32)

    # Sparse bounded target (~88% zeros, rest in [-1, 1]) — mimic real distribution
    target = np.zeros(n_rows, dtype=np.float32)
    nonzero_mask = rng.random(n_rows) > 0.88
    target[nonzero_mask] = rng.uniform(-1.0, 1.0, size=nonzero_mask.sum()).astype(np.float32)

    df = pd.DataFrame(feats, columns=[f"Feature_{i}" for i in range(n_feats)])
    df.insert(0, "moon", moons)
    df.insert(0, "id", np.arange(n_rows))
    y = pd.Series(target, name="target")

    with tempfile.TemporaryDirectory() as tmp:
        train(df, y, tmp)
        out = infer(df, tmp)

        assert list(out.columns) == ["id", "moon", "prediction"], (
            f"Bad column schema: {list(out.columns)}"
        )
        assert len(out) == n_rows, f"Row count mismatch: {len(out)} vs {n_rows}"
        assert out["prediction"].notna().all(), "NaN in predictions"
        print(
            f"OK - schema={list(out.columns)} rows={len(out)} "
            f"pred_range=[{out['prediction'].min():.4f}, {out['prediction'].max():.4f}]"
        )
