"""v28: v14 + per-moon feature z-scoring (cross-sectional normalization).

Each feature is z-scored within its moon → comparable across moons, regime-neutral.
v14 = 0.0497 with raw features. v28 tests if cross-sectional normalization helps XGB.
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _zscore_per_moon(X_df, feature_cols, moon_col, eps=1e-9):
    """Per-moon z-score for each feature."""
    out = X_df[[moon_col] + feature_cols].copy()
    # vectorized: groupby once, subtract mean, divide std
    mean = out.groupby(moon_col)[feature_cols].transform("mean")
    std = out.groupby(moon_col)[feature_cols].transform("std").replace(0, np.nan)
    out_z = (out[feature_cols] - mean) / (std + eps)
    out_z = out_z.fillna(0.0).astype(np.float32)
    return out_z


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    print(f"[v28/train] z-scoring {len(feature_cols)} features per moon")
    Xz = _zscore_per_moon(X_train, feature_cols, moon_col=moon_col)
    Xz[moon_col] = X_train[moon_col].values
    Xz["id"] = X_train["id" if "id" in X_train.columns else "Id"].values

    join_cols = [c for c in ("id", "moon") if c in Xz.columns and c in y_train.columns]
    merged = Xz.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)

    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v28/train] fit XGB on {len(Xt)} rows × {len(feature_cols)} z-scored feats")

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(Xt, y_ranked)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col, moon_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _, moon_col = pickle.load(f)
    Xz = _zscore_per_moon(X_test, feature_cols, moon_col=moon_col)
    preds = model.predict(Xz[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
