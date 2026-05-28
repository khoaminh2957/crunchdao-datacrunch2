"""CrunchDAO DataCrunch #2 — Ridge regression baseline (diversity + linear floor for ensemble).

Pipeline: SimpleImputer(median) -> StandardScaler -> Ridge(alpha=10.0)
Target: per-moon percentile rank (Spearman-native).
LGBM baseline reference: 0.0124 per-moon Spearman.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _rank_per_moon(y_df: pd.DataFrame, moon_col: str, target_col: str) -> pd.Series:
    """Per-moon percentile rank in (0,1] — matches Spearman scoring."""
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _build_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
):
    """Fit Ridge pipeline on per-moon-ranked target."""
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    print(f"[ridge_v1.train] X shape={X_train.shape}, y shape={y_train.shape}, "
          f"n_features={len(feature_cols)}")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col_raw = "target" if "target" in merged.columns else [
        c for c in y_train.columns if c not in ("id", "moon")
    ][0]

    mask = merged[target_col_raw].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    print(f"[ridge_v1.train] merged after dropna={merged.shape}, target='{target_col_raw}'")

    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(
        merged[[moon_col, target_col_raw]], moon_col=moon_col, target_col=target_col_raw
    )
    print(f"[ridge_v1.train] rank target: mean={float(y_ranked.mean()):.3f}, "
          f"std={float(y_ranked.std()):.3f}")

    Xt = merged[feature_cols]

    pipeline = _build_pipeline(alpha=10.0)
    print(f"[ridge_v1.train] fitting Ridge(alpha=10.0) on {len(Xt)} rows x {len(feature_cols)} features")
    pipeline.fit(Xt, y_ranked.values)

    os.makedirs(model_directory_path, exist_ok=True)
    out_path = f"{model_directory_path}/model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((pipeline, feature_cols), f)
    print(f"[ridge_v1.train] saved pipeline + {len(feature_cols)} feature_cols -> {out_path}")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
) -> pd.DataFrame:
    """Predict for test. Returns [id, moon, prediction] as required by scorer."""
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        pipeline, feature_cols = pickle.load(f)
    print(f"[ridge_v1.infer] X shape={X_test.shape}, n_features_trained={len(feature_cols)}")

    # Tolerate any missing trained features at infer time
    feature_cols_use = [c for c in feature_cols if c in X_test.columns]
    if len(feature_cols_use) != len(feature_cols):
        print(f"[ridge_v1.infer] WARN: {len(feature_cols) - len(feature_cols_use)} trained features absent in X_test")

    Xt = X_test[feature_cols_use]
    preds = pipeline.predict(Xt)

    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    out = pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds.astype(np.float32),
    })
    print(f"[ridge_v1.infer] returning {out.shape}, cols={list(out.columns)}")
    return out


if __name__ == "__main__":
    # Smoke test: 1000 synthetic rows, ~1150 features, ~88% zero target on [-1,1]
    import tempfile

    rng = np.random.default_rng(0)
    n_rows = 1000
    n_feat = 1150
    n_moons = 20

    feat_names = [f"f{i:04d}" for i in range(n_feat)]
    X = pd.DataFrame(rng.standard_normal((n_rows, n_feat)).astype(np.float32), columns=feat_names)
    X.insert(0, "moon", rng.integers(0, n_moons, size=n_rows))
    X.insert(0, "id", np.arange(n_rows))

    # ~88% zeros, rest uniform on [-1,1]
    y_vals = np.where(rng.random(n_rows) < 0.88, 0.0, rng.uniform(-1, 1, n_rows)).astype(np.float32)
    y = pd.DataFrame({"id": X["id"].values, "moon": X["moon"].values, "target": y_vals})

    with tempfile.TemporaryDirectory() as tmp:
        train(X, y, tmp)
        preds = infer(X.drop(columns=[]), tmp)

        # Schema assertions
        assert list(preds.columns) == ["id", "moon", "prediction"], f"bad cols: {list(preds.columns)}"
        assert len(preds) == n_rows, f"row count {len(preds)} != {n_rows}"
        assert preds["prediction"].notna().all(), "NaN in predictions"
        assert preds["id"].is_unique, "duplicated id"
        assert set(preds["moon"].unique()).issubset(set(range(n_moons))), "unexpected moon values"

        # Pickle sanity: round-trip object types
        with open(f"{tmp}/model.pkl", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, tuple) and len(obj) == 2, "pickle must be (pipeline, feature_cols)"
        assert isinstance(obj[0], Pipeline), "first pickled element must be sklearn Pipeline"
        assert isinstance(obj[1], list) and len(obj[1]) == n_feat, "feature_cols mismatch"

        print(f"[smoke] OK  rows={len(preds)}  pred mean={preds['prediction'].mean():.4f}  "
              f"std={preds['prediction'].std():.4f}  "
              f"min={preds['prediction'].min():.4f}  max={preds['prediction'].max():.4f}")
