"""CrunchDAO DataCrunch #2 — v6 ENSEMBLE entrypoint (LGBM + XGB + CatBoost + Ridge + ExtraTrees + DART).

Thin wrapper that delegates to models/ensemble_v2.py. The ensemble itself
gracefully skips any member whose library isn't installed in the cloud env,
so a missing dep degrades to fewer members rather than a hard fail.

Each member trains on the same per-moon-ranked target; infer() averages
per-moon-ranked predictions across all successful members → [id, moon, prediction].
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup so models.* and the sibling lgbm baseline import correctly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
import sys as _sys
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# LGBM baseline (used as the "lgbm" ensemble member, also fallback if ensemble fails)
# ---------------------------------------------------------------------------
def _load_top_features(model_directory_path: str):
    here = Path(__file__).resolve().parent
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


def _lgbm_train(X_train, y_train, model_directory_path):
    from lightgbm import LGBMRegressor
    feature_cols_all = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    top = _load_top_features(model_directory_path)
    feature_cols = [c for c in top if c in feature_cols_all] if top else feature_cols_all
    if len(feature_cols) < 50:
        feature_cols = feature_cols_all
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col_raw = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id", "moon")][0]
    mask = merged[target_col_raw].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(merged[[moon_col, target_col_raw]], moon_col=moon_col, target_col=target_col_raw)
    sample_w = _recency_weights(merged[moon_col].values, tau=50.0)
    Xt = merged[feature_cols]
    print(f"[lgbm] training {len(Xt)} rows × {len(feature_cols)} feats")
    model = LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=63,
        min_data_in_leaf=200, colsample_bytree=0.5, reg_lambda=2.0,
        verbose=-1, n_jobs=-1,
    )
    model.fit(Xt, y_ranked, sample_weight=sample_w)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col_raw), f)


def _lgbm_infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _ = pickle.load(f)
    feature_cols = [c for c in feature_cols if c in X_test.columns]
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds,
    })


# ---------------------------------------------------------------------------
# Public entrypoints — try ensemble first, fall back to LGBM baseline
# ---------------------------------------------------------------------------
def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str):
    """Train ensemble (LGBM+XGB+CatBoost+Ridge+ExtraTrees+DART). Members whose deps are missing skip gracefully."""
    print(f"[main/train] X={X_train.shape}, y={y_train.shape}")
    try:
        from models.ensemble_v2 import train as ens_train  # noqa
        ens_train(X_train, y_train, model_directory_path)
        # Mark which path was used so infer picks it up
        with open(f"{model_directory_path}/_mode.txt", "w") as f:
            f.write("ensemble")
        print("[main/train] ensemble path SUCCESS")
        return
    except Exception as e:
        print(f"[main/train] ensemble path FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print("[main/train] falling back to LGBM baseline")
    _lgbm_train(X_train, y_train, model_directory_path)
    with open(f"{model_directory_path}/_mode.txt", "w") as f:
        f.write("lgbm")


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    """Dispatch based on _mode.txt written by train()."""
    mode_file = Path(model_directory_path) / "_mode.txt"
    mode = mode_file.read_text().strip() if mode_file.exists() else "lgbm"
    print(f"[main/infer] mode={mode}, X={X_test.shape}")
    if mode == "ensemble":
        try:
            from models.ensemble_v2 import infer as ens_infer
            out = ens_infer(X_test, model_directory_path)
            print(f"[main/infer] ensemble returned {out.shape}, cols={list(out.columns)}")
            return out
        except Exception as e:
            print(f"[main/infer] ensemble FAILED: {type(e).__name__}: {e} — falling back to lgbm")
    out = _lgbm_infer(X_test, model_directory_path)
    print(f"[main/infer] lgbm returned {out.shape}, cols={list(out.columns)}")
    return out
