"""
CrunchDAO DataCrunch #2 - XGBoost v1 (xgboost>=2.0)

Different model family from LGBM baseline (run #81537, sub #4, score 0.0124).
Trains XGBRegressor on rank-per-moon transformed target (Spearman-native).

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
import xgboost as xgb


MODEL_FILENAME = "model.pkl"

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.5,
    tree_method="hist",
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return Feature_* columns in stable sorted order."""
    feats = [c for c in df.columns if c.startswith("Feature_")]
    # Sort by numeric suffix to be robust to lexicographic ordering
    feats.sort(key=lambda c: int(c.split("_", 1)[1]))
    return feats


def _rank_per_moon(y: pd.Series, moons: pd.Series) -> np.ndarray:
    """Rank target within each moon, scaled to [-1, 1].

    Spearman-native transform: training on per-group ranks aligns the
    regression loss with the competition's per-moon Spearman scoring.
    Ties are averaged (pandas default).
    """
    s = pd.Series(np.asarray(y, dtype=float), index=moons.index)
    ranked = s.groupby(moons).transform(
        lambda g: g.rank(method="average", pct=True)
    )
    # pct rank in (0, 1] -> recenter to [-1, 1]
    return (ranked.to_numpy() * 2.0) - 1.0


# ---------------------------------------------------------------------------
# train / infer
# ---------------------------------------------------------------------------
def train(X_train: pd.DataFrame, y_train, model_directory_path: str) -> None:
    """Fit XGBRegressor on rank-per-moon target and pickle the artifact."""
    os.makedirs(model_directory_path, exist_ok=True)

    if "moon" not in X_train.columns:
        raise ValueError("X_train must contain a 'moon' column for per-moon ranking.")

    # y may be Series, ndarray, or single-column DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_series = y_train.iloc[:, 0]
    else:
        y_series = pd.Series(np.asarray(y_train), index=X_train.index)

    feature_cols = _feature_cols(X_train)
    if not feature_cols:
        raise ValueError("No Feature_* columns found in X_train.")

    y_ranked = _rank_per_moon(y_series, X_train["moon"])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train[feature_cols].to_numpy(dtype=np.float32), y_ranked)

    artifact = {"model": model, "feature_cols": feature_cols}
    with open(os.path.join(model_directory_path, MODEL_FILENAME), "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    """Score X_test and return DataFrame[id, moon, prediction].

    Column schema is load-bearing: the CrunchDAO scorer rejects any other
    layout (4 prior submissions failed before this was nailed down).
    """
    with open(os.path.join(model_directory_path, MODEL_FILENAME), "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols: List[str] = artifact["feature_cols"]

    missing = [c for c in feature_cols if c not in X_test.columns]
    if missing:
        raise ValueError(f"X_test missing {len(missing)} expected features, e.g. {missing[:3]}")

    preds = model.predict(X_test[feature_cols].to_numpy(dtype=np.float32))

    if "id" not in X_test.columns:
        raise ValueError("X_test must contain an 'id' column.")
    if "moon" not in X_test.columns:
        raise ValueError("X_test must contain a 'moon' column.")

    out = pd.DataFrame(
        {
            "id": X_test["id"].to_numpy(),
            "moon": X_test["moon"].to_numpy(),
            "prediction": preds.astype(np.float64),
        }
    )
    # Final guard: exact column order required by the scorer
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
        print(f"OK - schema={list(out.columns)} rows={len(out)} "
              f"pred_range=[{out['prediction'].min():.4f}, {out['prediction'].max():.4f}]")
