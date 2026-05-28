"""v50: SWEEP WINNER from GPU local CV — XGB n_est=700, depth=7, lr=0.05.

Sweep results (5-fold walk-forward CV on local data):
- v50 (700/d7/lr=0.05): 0.07108 ★ — WINNER (predicts ~0.064 cloud)
- vs v30 cloud (500/d6/lr=0.03 on raw+z): 0.0563
- vs v14 cloud (500/d6/lr=0.03 on raw): 0.0497

Single XGB, rank target, all 1150 features (no z-score — EDA showed redundant).
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    print(f"[v50/train] X={X_train.shape}, features={len(feature_cols)}")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v50/train] XGB(700/d7/lr0.05) on {len(Xt)} rows × {len(feature_cols)} feats — sweep winner")

    model = XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=1.0,
        reg_lambda=1.0,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(Xt, y_ranked)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _ = pickle.load(f)
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
