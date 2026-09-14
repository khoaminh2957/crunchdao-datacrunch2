"""v35: XGB(raw+z) + LGBM(raw+z), raw prediction average.

v30 XGB(raw+z) = 0.0563 ★
v35 trains LGBM same features, averages RAW preds (not ranks like buggy v17).
Different algos on same enriched features = decorrelated ⇒ ensemble lift.
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


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    raw_feats = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    id_col = "id" if "id" in X_train.columns else "Id"

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

    print(f"[v35/train] === fitting XGB ===")
    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    xgb.fit(Xt, y_ranked)
    print(f"[v35/train] === fitting LGBM ===")
    lgbm = LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=63,
        min_data_in_leaf=200, colsample_bytree=0.5, reg_lambda=2.0,
        verbose=-1, n_jobs=-1,
    )
    lgbm.fit(Xt, y_ranked)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((xgb, lgbm, feature_cols, raw_feats, target_col, moon_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        xgb, lgbm, feature_cols, raw_feats, _, moon_col = pickle.load(f)
    Xz = _zscore_per_moon(X_test, raw_feats, moon_col=moon_col)
    Xz.columns = [f"{c}_z" for c in raw_feats]
    Xc = pd.concat([X_test[raw_feats].reset_index(drop=True),
                    Xz.reset_index(drop=True)], axis=1)
    p_xgb = xgb.predict(Xc[feature_cols])
    p_lgbm = lgbm.predict(Xc[feature_cols])
    preds = (p_xgb + p_lgbm) / 2.0
    print(f"[v35/infer] XGB range=[{p_xgb.min():.4f}, {p_xgb.max():.4f}], LGBM range=[{p_lgbm.min():.4f}, {p_lgbm.max():.4f}]")
    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
