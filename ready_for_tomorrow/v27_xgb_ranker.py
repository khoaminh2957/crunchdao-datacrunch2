"""v27: XGBRanker (pairwise rank loss) — Spearman-native loss instead of MSE.

v14 uses XGBRegressor MSE on rank target. v27 uses XGBRanker rank:pairwise
which natively optimizes pairwise ordering — closer to Spearman scoring.
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRanker
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"

    # XGBRanker requires sort by group + group sizes
    merged = merged.sort_values(moon_col).reset_index(drop=True)
    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    # Discretize rank to integer relevance grades [0..31] for ranking loss
    y_relevance = (y_ranked * 32).clip(0, 31).astype(int)
    Xt = merged[feature_cols]
    group_sizes = merged.groupby(moon_col).size().values
    print(f"[v27/train] XGBRanker on {len(Xt)} rows, {len(group_sizes)} moon-groups (rank:pairwise loss)")

    model = XGBRanker(
        objective='rank:pairwise',
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(Xt, y_relevance, group=group_sizes)
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
