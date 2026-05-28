"""
train_baseline.py — competition-agnostic baseline.

Strategy:
  1. Load X_train / y_train from data/ (auto-detected from common CrunchDAO layouts).
  2. Fit a LightGBM + Ridge ensemble (50/50 blend by default).
  3. Use time-series-aware CV when a date/era column is present, else KFold.
  4. Write predictions on X_test to predictions.parquet at project root.

This is a STARTING POINT, not a winning model. Iterate on features after first submission.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Data discovery — CrunchDAO competitions vary in file naming.
# Try common conventions, then fall back to the largest parquet/csv in data/.
# ---------------------------------------------------------------------------

CANDIDATE_TRAIN_X = ["X_train.parquet", "X_train.csv", "train.parquet", "train.csv"]
CANDIDATE_TRAIN_Y = ["y_train.parquet", "y_train.csv", "target.parquet", "target.csv"]
CANDIDATE_TEST_X = ["X_test.parquet", "X_test.csv", "test.parquet", "test.csv"]


def _find(data_dir: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _detect_id_and_date(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Return (id_col, date_col) if present in df."""
    id_col = None
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("id", "row_id", "sample_id", "obs_id") and id_col is None:
            id_col = c
        if lc in ("date", "era", "moon", "moon_id", "period", "time") and date_col is None:
            date_col = c
    return id_col, date_col


def main() -> int:
    cwd = Path.cwd()
    data_dir = cwd / "data"
    if not data_dir.exists():
        print(f"[ERROR] data/ not found. Run download_data.py first.", file=sys.stderr)
        return 2

    x_train_path = _find(data_dir, CANDIDATE_TRAIN_X)
    y_train_path = _find(data_dir, CANDIDATE_TRAIN_Y)
    x_test_path = _find(data_dir, CANDIDATE_TEST_X)

    if not (x_train_path and y_train_path and x_test_path):
        print("[ERROR] Could not auto-detect train/test files in data/.", file=sys.stderr)
        print(f"        Looked for: {CANDIDATE_TRAIN_X}, {CANDIDATE_TRAIN_Y}, {CANDIDATE_TEST_X}", file=sys.stderr)
        print(f"        Found in data/: {sorted(p.name for p in data_dir.iterdir())}", file=sys.stderr)
        print("        Edit train_baseline.py CANDIDATE_* lists to match this competition.", file=sys.stderr)
        return 3

    print(f"[train] loading {x_train_path.name} / {y_train_path.name} / {x_test_path.name}")
    X_train = _load(x_train_path)
    y_train_df = _load(y_train_path)
    X_test = _load(x_test_path)

    id_col, date_col = _detect_id_and_date(X_train)
    print(f"[train] detected id_col={id_col!r} date_col={date_col!r}")

    # Align target — assume y has [id_col, target] or just [target]
    target_col = None
    for c in y_train_df.columns:
        if c.lower() in ("y", "target", "label", "return"):
            target_col = c
            break
    if target_col is None:
        target_col = y_train_df.columns[-1]
    y = y_train_df[target_col].values
    print(f"[train] target_col={target_col!r}, n_train={len(y)}, n_test={len(X_test)}")

    # Build feature matrix — drop id + date, keep numeric only
    feature_cols = [
        c for c in X_train.columns
        if c not in (id_col, date_col) and pd.api.types.is_numeric_dtype(X_train[c])
    ]
    print(f"[train] using {len(feature_cols)} numeric features")
    X = X_train[feature_cols].fillna(0.0).values
    X_te = X_test[feature_cols].fillna(0.0).values

    # CV — TimeSeriesSplit if we have an ordered date column, else KFold
    if date_col is not None and date_col in X_train.columns:
        order = X_train[date_col].argsort().values
        X = X[order]
        y = y[order]
        cv = TimeSeriesSplit(n_splits=5)
        print(f"[train] using TimeSeriesSplit on {date_col}")
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print("[train] using KFold (no date column detected)")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    oof = np.zeros(len(y))
    test_preds_lgb = np.zeros(len(X_te))
    test_preds_ridge = np.zeros(len(X_te))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # LightGBM
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        model = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=1500,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_preds_lgb += model.predict(X_te, num_iteration=model.best_iteration) / cv.get_n_splits()

        # Ridge as a stable low-variance baseline blend
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_tr)
        test_preds_ridge += ridge.predict(X_te) / cv.get_n_splits()

        fold_rmse = np.sqrt(mean_squared_error(y_va, oof[va_idx]))
        print(f"[train] fold {fold + 1} RMSE={fold_rmse:.5f}")

    oof_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"[train] OOF RMSE (LGB only) = {oof_rmse:.5f}")

    # 70/30 LGB-Ridge ensemble — adjust after first leaderboard read
    test_preds = 0.7 * test_preds_lgb + 0.3 * test_preds_ridge

    # Write submission
    out = pd.DataFrame()
    if id_col is not None and id_col in X_test.columns:
        out[id_col] = X_test[id_col].values
    if date_col is not None and date_col in X_test.columns:
        out[date_col] = X_test[date_col].values
    out["prediction"] = test_preds

    out_path = cwd / "predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[train] wrote {out_path} — {len(out)} rows, cols={list(out.columns)}")

    # Snapshot metadata for later audit
    meta = {
        "oof_rmse": float(oof_rmse),
        "n_train": int(len(y)),
        "n_test": int(len(X_te)),
        "n_features": int(len(feature_cols)),
        "id_col": id_col,
        "date_col": date_col,
        "target_col": target_col,
        "blend": "0.7*lgb + 0.3*ridge",
    }
    with open(cwd / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] wrote train_meta.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
