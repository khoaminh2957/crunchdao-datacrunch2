"""v31: raw + per-moon z-score + per-moon rank features (3 views = 3450 feats).

v14 raw only = 0.0497
v28 z-score only = 0.0512
v30 raw + z-score = 0.0563 ★
v31 adds rank features (per-moon) on top → 3 views per feature.
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon_series(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _zscore_per_moon(X_df, feature_cols, moon_col, eps=1e-9):
    out = X_df[[moon_col] + feature_cols].copy()
    mean = out.groupby(moon_col)[feature_cols].transform("mean")
    std = out.groupby(moon_col)[feature_cols].transform("std").replace(0, np.nan)
    return ((out[feature_cols] - mean) / (std + eps)).fillna(0.0).astype(np.float32)


def _rank_per_moon_features(X_df, feature_cols, moon_col):
    n = X_df.groupby(moon_col)[feature_cols].transform("count")
    return (X_df.groupby(moon_col)[feature_cols].rank(method="average") / n).fillna(0.5).astype(np.float32)


def _build_combined(X_df, raw_feats, moon_col):
    """Returns DataFrame with raw + _z + _r columns."""
    Xz = _zscore_per_moon(X_df, raw_feats, moon_col=moon_col)
    Xz.columns = [f"{c}_z" for c in raw_feats]
    Xr = _rank_per_moon_features(X_df, raw_feats, moon_col=moon_col)
    Xr.columns = [f"{c}_r" for c in raw_feats]
    return pd.concat([X_df[raw_feats].reset_index(drop=True),
                      Xz.reset_index(drop=True),
                      Xr.reset_index(drop=True)], axis=1)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    raw_feats = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    id_col = "id" if "id" in X_train.columns else "Id"
    print(f"[v31/train] building 3-view features: raw + z + r = {len(raw_feats)*3} cols")

    Xc = _build_combined(X_train, raw_feats, moon_col=moon_col)
    Xc[id_col] = X_train[id_col].values
    Xc[moon_col] = X_train[moon_col].values

    feature_cols = raw_feats + [f"{c}_z" for c in raw_feats] + [f"{c}_r" for c in raw_feats]
    merged = Xc.merge(y_train, on=[c for c in ("id","moon") if c in Xc.columns and c in y_train.columns], how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    y_ranked = _rank_per_moon_series(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v31/train] fit XGB on {len(Xt)} rows × {len(feature_cols)} feats")

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.4,  # reduce colsample bc 3x features
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(Xt, y_ranked)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, raw_feats, target_col, moon_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, raw_feats, _, moon_col = pickle.load(f)
    Xc = _build_combined(X_test, raw_feats, moon_col=moon_col)
    preds = model.predict(Xc[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
