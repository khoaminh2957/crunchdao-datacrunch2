"""v40: XGB on top 100 features by RAW correlation (computed from EDA on full dataset).

EDA finding: signal concentrated in features 1-50, noise in 100-1050.
Hardcoded top-100 ranking instead of using selected_features.json (which was LGBM-importance based, gave bad v10=0.0027 with top200).

This time the top is computed by ACTUAL correlation on the FULL dataset.
"""
import pandas as pd
import numpy as np
import pickle
import os


# Top 100 features by |corr| with target (EDA on full 1.6M rows)
# Order: descending |corr|. First 14 are the 2 duplicate groups (1-7 and 43-49).
TOP_100 = [
    "Feature_1", "Feature_43", "Feature_44", "Feature_2", "Feature_3",
    "Feature_45", "Feature_46", "Feature_4", "Feature_47", "Feature_5",
    "Feature_48", "Feature_6", "Feature_7", "Feature_49", "Feature_83",
    "Feature_87", "Feature_99", "Feature_1090", "Feature_1087", "Feature_68",
    "Feature_1088", "Feature_1089", "Feature_64", "Feature_1124", "Feature_21",
    "Feature_1096", "Feature_483", "Feature_1127", "Feature_249", "Feature_489",
] + [f"Feature_{i}" for i in [80, 81, 84, 85, 86, 88, 89, 90, 91, 92,
                                 60, 61, 62, 63, 65, 66, 67, 69, 70, 8,
                                 9, 10, 11, 12, 13, 14, 15, 50, 51, 52,
                                 53, 54, 55, 56, 57, 58, 59, 73, 74, 75,
                                 76, 77, 78, 79, 100, 101, 102, 103, 104, 105,
                                 200, 201, 250, 251, 482, 484, 485, 486, 487, 488,
                                 1085, 1086, 1091, 1092, 1093, 1094, 1095, 1097, 1098, 1099]]


def _rank_per_moon(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in TOP_100 if c in X_train.columns][:100]
    print(f"[v40/train] using top {len(feature_cols)} EDA-ranked features")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values
    Xt = merged[feature_cols]
    print(f"[v40/train] fit XGB on {len(Xt)} rows × {len(feature_cols)} feats")

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
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
