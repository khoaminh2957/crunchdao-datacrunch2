"""v20: rank + XGB(500/d6/lr0.03) + top500 features.

top200 hurt LGBM (v10 = 0.0027 vs 0.0124 baseline). Maybe top200 too aggressive
for XGB too — try top500 instead (50% reduction vs 80%).
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path


def _rank_per_moon(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _load_top_features():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "selected_features.json",
        here / "improvements" / "selected_features.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                top = data.get("top_500_combined") or data.get("top_200_combined") or data.get("top_100_combined") or []
                return list(top)
            except Exception:
                pass
    return None


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols_all = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    top = _load_top_features()
    if top:
        top500 = list(top)[:500]
        feature_cols = [c for c in top500 if c in feature_cols_all]
        if len(feature_cols) < 100:
            feature_cols = feature_cols_all
            print(f"[v20] top overlap small ({len(feature_cols)}); using all")
        else:
            print(f"[v20] using top {len(feature_cols)} features (from {len(feature_cols_all)})")
    else:
        feature_cols = feature_cols_all

    print(f"[v20/train] X={X_train.shape}, using {len(feature_cols)} feats")
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"

    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(Xt, y_ranked)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        model, feature_cols, _ = pickle.load(f)
    feature_cols = [c for c in feature_cols if c in X_test.columns]
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
