"""v29: v14 + per-moon RANK features (each feature replaced by per-moon percentile rank).

v28 per-moon z-score = 0.0512 (beat v14 0.0497).
v29 tests per-moon RANK transform — more aggressive than z-score, eliminates outliers.
Scoring is Spearman = rank-correlation, so rank features align with target metric.
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon_series(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _rank_per_moon_features(X_df, feature_cols, moon_col):
    """Per-moon percentile rank for each feature."""
    out = X_df[feature_cols].copy()
    n = X_df.groupby(moon_col)[feature_cols].transform("count")
    out_r = X_df.groupby(moon_col)[feature_cols].rank(method="average") / n
    return out_r.fillna(0.5).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    print(f"[v29/train] per-moon RANK transform on {len(feature_cols)} features")
    Xr = _rank_per_moon_features(X_train, feature_cols, moon_col=moon_col)
    Xr[moon_col] = X_train[moon_col].values
    Xr["id"] = X_train["id" if "id" in X_train.columns else "Id"].values

    join_cols = [c for c in ("id", "moon") if c in Xr.columns and c in y_train.columns]
    merged = Xr.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    y_ranked = _rank_per_moon_series(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v29/train] fit XGB on {len(Xt)} rows × {len(feature_cols)} per-moon-rank feats")

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
    Xr = _rank_per_moon_features(X_test, feature_cols, moon_col=moon_col)
    preds = model.predict(Xr[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
