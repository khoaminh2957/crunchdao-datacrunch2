"""v30: v14 + COMBINED raw + per-moon z-score features (2300 total).

v14 raw = 0.0497
v28 z-score only = 0.0512 ★
v30 = both views (raw + z-score suffix) → XGB picks per split.
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
    out_z = (out[feature_cols] - mean) / (std + eps)
    return out_z.fillna(0.0).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    raw_feats = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    id_col = "id" if "id" in X_train.columns else "Id"
    print(f"[v30/train] raw={len(raw_feats)} + z-score = {len(raw_feats)*2} feats")

    Xz = _zscore_per_moon(X_train, raw_feats, moon_col=moon_col)
    Xz.columns = [f"{c}_z" for c in raw_feats]
    X_combined = pd.concat([X_train[raw_feats].reset_index(drop=True),
                            Xz.reset_index(drop=True)], axis=1)
    X_combined[id_col] = X_train[id_col].values
    X_combined[moon_col] = X_train[moon_col].values

    feature_cols = raw_feats + [f"{c}_z" for c in raw_feats]

    merged = X_combined.merge(y_train, on=[c for c in ("id","moon") if c in X_combined.columns and c in y_train.columns], how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    y_ranked = _rank_per_moon_series(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v30/train] fit XGB on {len(Xt)} rows × {len(feature_cols)} feats")

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(Xt, y_ranked)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, raw_feats, target_col, moon_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, raw_feats, _, moon_col = pickle.load(f)
    Xz = _zscore_per_moon(X_test, raw_feats, moon_col=moon_col)
    Xz.columns = [f"{c}_z" for c in raw_feats]
    X_combined = pd.concat([X_test[raw_feats].reset_index(drop=True),
                            Xz.reset_index(drop=True)], axis=1)
    preds = model.predict(X_combined[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
