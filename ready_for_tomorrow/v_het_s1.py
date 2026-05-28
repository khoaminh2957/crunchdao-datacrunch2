"""v_het_s1: single-seed hetero ensemble, seed=1 (different from v68's 42). Lottery shot."""
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
    mask = merged[target_col].notna(); merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values; Xt = merged[feature_cols].values.astype(np.float32)
    m_xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.5, tree_method='hist', n_jobs=-1, random_state=1, verbosity=0)
    m_xgb.fit(Xt, yt)
    m_lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                            colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, n_jobs=-1, random_state=1)
    m_lgbm.fit(Xt, yt)
    imp = SimpleImputer(strategy='median'); Xt_imp = imp.fit_transform(Xt)
    m_ridge = Ridge(alpha=10.0, random_state=1); m_ridge.fit(Xt_imp, yt)
    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((m_xgb, m_lgbm, m_ridge, imp, feature_cols, target_col), f)

def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        m_xgb, m_lgbm, m_ridge, imp, feature_cols, _ = pickle.load(f)
    Xt = X_test[feature_cols].values.astype(np.float32)
    p_ens = (2*m_xgb.predict(Xt) + 1*m_lgbm.predict(Xt) + 0.5*m_ridge.predict(imp.transform(Xt))) / 3.5
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": p_ens})
