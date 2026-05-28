"""v24: seed-bagging XGB — average 3 XGB models with different seeds.

v17 ensemble bug was rank-averaging LGBM+XGB. v24 averages RAW predictions of
3 same-arch XGBs with different random seeds. Should be more stable than v14
single seed at marginal cost.
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
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]

    models = []
    for seed in [42, 7, 2026]:
        print(f"[v24/train] fitting XGB seed={seed}")
        m = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5,
            tree_method='hist', n_jobs=-1, random_state=seed, verbosity=0,
        )
        m.fit(Xt, y_ranked)
        models.append(m)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((models, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        models, feature_cols, _ = pickle.load(f)
    Xt = X_test[feature_cols]
    preds = np.mean([m.predict(Xt) for m in models], axis=0)
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
