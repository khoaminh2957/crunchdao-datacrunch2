"""LightGBM member of the ensemble — standalone train/infer.

Same logic that was inline in main.py (rank target + recency weights tau=50
+ top-200 features filter). Extracted so ensemble_v2 can import it without
recursing back into main.py (which is the ensemble entrypoint).
"""
from __future__ import annotations
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _load_top_features(model_directory_path: str):
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "improvements" / "selected_features.json",
        Path(model_directory_path) / "selected_features.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                top = data.get("top_200_combined") or data.get("top_100_combined") or []
                if top:
                    return list(top)
            except Exception:
                pass
    return None


def _rank_per_moon(y_df: pd.DataFrame, moon_col: str, target_col: str) -> pd.Series:
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _recency_weights(moons: np.ndarray, tau: float = 50.0) -> np.ndarray:
    mmax = float(np.max(moons))
    w = np.exp(-(mmax - moons.astype(np.float64)) / tau)
    return (w / w.mean()).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from lightgbm import LGBMRegressor

    feature_cols_all = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    top = _load_top_features(model_directory_path)
    feature_cols = [c for c in top if c in feature_cols_all] if top else feature_cols_all
    if len(feature_cols) < 50:
        feature_cols = feature_cols_all
    print(f"[lgbm_v1] features={len(feature_cols)}, X={X_train.shape}, y={y_train.shape}")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col_raw = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id", "moon")][0]
    mask = merged[target_col_raw].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"

    y_ranked = _rank_per_moon(merged[[moon_col, target_col_raw]], moon_col=moon_col, target_col=target_col_raw)
    sample_w = _recency_weights(merged[moon_col].values, tau=50.0)
    Xt = merged[feature_cols]
    print(f"[lgbm_v1] fit on {len(Xt)} rows × {len(feature_cols)} feats; rank mean={float(y_ranked.mean()):.3f}")

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=200,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(Xt, y_ranked, sample_weight=sample_w)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col_raw), f)
    print(f"[lgbm_v1] saved → {model_directory_path}/model.pkl")


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _ = pickle.load(f)
    feature_cols = [c for c in feature_cols if c in X_test.columns]
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    out = pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds,
    })
    print(f"[lgbm_v1.infer] returning {out.shape}, cols={list(out.columns)}")
    return out


if __name__ == "__main__":
    np.random.seed(0)
    moons = np.repeat(np.arange(1, 11), 100)
    ids = np.tile(np.arange(100), 10)
    X = pd.DataFrame({"id": ids, "moon": moons})
    for i in [718, 396, 514, 105, 246]:
        X[f"Feature_{i}"] = np.random.randn(len(X))
    y = pd.DataFrame({"id": ids, "moon": moons,
                      "target": np.where(np.random.rand(len(X)) > 0.85, np.random.randn(len(X)) * 0.3, 0)})
    train(X, y, "/tmp/lgbm_v1_smoke")
    out = infer(X, "/tmp/lgbm_v1_smoke")
    assert list(out.columns) == ["id", "moon", "prediction"], out.columns
    print("[lgbm_v1] smoke OK")
