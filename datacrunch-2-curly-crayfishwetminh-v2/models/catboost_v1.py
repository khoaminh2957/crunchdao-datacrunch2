"""CrunchDAO DataCrunch #2 — CatBoost v1 with rank target. requires catboost>=1.2"""
import os
import pickle

import numpy as np
import pandas as pd


def _rank_per_moon(y_df: pd.DataFrame, moon_col: str, target_col: str) -> pd.Series:
    """Per-moon percentile rank, Spearman-native target."""
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
):
    """Train CatBoost regressor on per-moon-ranked target for ensemble diversity vs LGBM."""
    from catboost import CatBoostRegressor

    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    print(f"[train] X shape={X_train.shape}, y shape={y_train.shape}, n_features={len(feature_cols)}")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col_raw = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id", "moon")][0]
    print(f"[train] target='{target_col_raw}', merged={merged.shape}")

    mask = merged[target_col_raw].notna()
    merged = merged.loc[mask].reset_index(drop=True)

    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(merged[[moon_col, target_col_raw]], moon_col=moon_col, target_col=target_col_raw)
    print(f"[train] rank target: mean={float(y_ranked.mean()):.3f}, std={float(y_ranked.std()):.3f}")

    Xt = merged[feature_cols]
    print(f"[train] fitting CatBoost on {len(Xt)} rows × {len(feature_cols)} features")

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Xt, y_ranked)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col_raw), f)
    print(f"[train] saved CatBoost model + {len(feature_cols)} features (target was rank-transformed)")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
) -> pd.DataFrame:
    """Predict for test set. Scorer requires columns [id, moon, prediction]."""
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _target_col_raw = pickle.load(f)
    print(f"[infer] X shape={X_test.shape}, cols sample={list(X_test.columns)[:5]}")

    feature_cols = [c for c in feature_cols if c in X_test.columns]
    Xt = X_test[feature_cols]
    preds = model.predict(Xt)

    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    out = pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds,
    })
    print(f"[infer] returning {out.shape}, cols={list(out.columns)}")
    return out


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(42)
    n_rows = 1000
    n_features = 1150
    n_moons = 10

    feat_cols = [f"Feature_{i}" for i in range(n_features)]
    moons = rng.integers(0, n_moons, size=n_rows)
    ids = np.arange(n_rows)
    X = pd.DataFrame(rng.standard_normal((n_rows, n_features)).astype(np.float32), columns=feat_cols)
    X.insert(0, "moon", moons)
    X.insert(0, "id", ids)

    y_vals = rng.uniform(-1, 1, size=n_rows).astype(np.float32)
    zero_mask = rng.random(n_rows) < 0.88
    y_vals[zero_mask] = 0.0
    y = pd.DataFrame({"id": ids, "moon": moons, "target": y_vals})

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[smoke] training in {tmp}")
        train(X, y, tmp)
        out = infer(X, tmp)
        assert list(out.columns) == ["id", "moon", "prediction"], f"bad cols: {out.columns.tolist()}"
        assert len(out) == n_rows
        print(f"[smoke] OK: out shape={out.shape}, pred range=[{out['prediction'].min():.4f}, {out['prediction'].max():.4f}]")
