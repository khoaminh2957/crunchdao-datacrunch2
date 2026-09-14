"""v63: 3-seed ensemble of v14 (XGB 500/d6/lr=0.03 RAW, seeds 42, 7, 2026, raw pred avg)."""
import pandas as pd, numpy as np, pickle, os

def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values
    Xt = merged[feature_cols]
    models = []
    for seed in [42, 7, 2026]:
        print(f"[v63/train] XGB seed={seed}")
        m = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
            tree_method='hist', n_jobs=-1, random_state=seed, verbosity=0,
        )
        m.fit(Xt, yt)
        models.append(m)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((models, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        models, feature_cols, _ = pickle.load(f)
    preds = np.mean([m.predict(X_test[feature_cols]) for m in models], axis=0)
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
