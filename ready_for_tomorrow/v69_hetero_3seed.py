"""v69: 3-seed hetero ensemble. (XGB+LGBM+Ridge) × seeds [42, 7, 2026], raw pred avg.

v68 = 0.0761 cloud (single-seed hetero, predicted 0.067).
v69 should converge closer to true mean ~0.067 with lower variance.
"""
import pandas as pd, numpy as np, pickle, os


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values
    Xt = merged[feature_cols].values.astype(np.float32)

    seeds = [42, 7, 2026]
    xgbs, lgbms = [], []
    for s in seeds:
        print(f"[v69/train] XGB seed={s}")
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.5,
                         tree_method='hist', n_jobs=-1, random_state=s, verbosity=0)
        m.fit(Xt, yt); xgbs.append(m)
        print(f"[v69/train] LGBM seed={s}")
        m = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                         min_data_in_leaf=200, colsample_bytree=0.5, reg_lambda=2.0,
                         verbose=-1, n_jobs=-1, random_state=s)
        m.fit(Xt, yt); lgbms.append(m)

    imp = SimpleImputer(strategy='median')
    Xt_imp = imp.fit_transform(Xt)
    ridge = Ridge(alpha=10.0, random_state=42); ridge.fit(Xt_imp, yt)
    print("[v69/train] Ridge done")

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((xgbs, lgbms, ridge, imp, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        xgbs, lgbms, ridge, imp, feature_cols, _ = pickle.load(f)
    Xt = X_test[feature_cols].values.astype(np.float32)
    p_xgb = np.mean([m.predict(Xt) for m in xgbs], axis=0)
    p_lgbm = np.mean([m.predict(Xt) for m in lgbms], axis=0)
    p_ridge = ridge.predict(imp.transform(Xt))
    p_ens = (2 * p_xgb + 1 * p_lgbm + 0.5 * p_ridge) / 3.5
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": p_ens})
