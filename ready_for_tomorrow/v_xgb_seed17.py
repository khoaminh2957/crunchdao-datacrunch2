"""Single XGB seed=17 (lottery shot, predicted cloud 0.065 ± 0.038)."""
import pandas as pd, numpy as np, pickle, os

def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna(); merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values; Xt = merged[feature_cols]
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                     colsample_bytree=0.5, tree_method='hist', n_jobs=-1, random_state=17, verbosity=0)
    m.fit(Xt, yt)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f: pickle.dump((m, feature_cols, target_col), f)

def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f: m, feature_cols, _ = pickle.load(f)
    preds = m.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
