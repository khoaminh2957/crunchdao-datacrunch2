"""v76: 2-stage model.
Stage 1: XGBClassifier P(non-zero)
Stage 2: XGBRegressor on non-zero only
Combined: pred = P(non-zero) * regressor_pred.

Local R4 validator: lift +0.024 mean across moons 772-781. Mixed at k=9.
"""
import pandas as pd, numpy as np, pickle, os


def train(X_train, y_train, model_directory_path):
    from xgboost import XGBRegressor, XGBClassifier
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values.astype(np.float32)
    Xt = merged[feature_cols].values.astype(np.float32)
    print(f"[v76/train] 2-stage on {len(Xt)} rows × {len(feature_cols)} feats")

    # Stage 1: classify non-zero
    y_nz = (np.abs(yt) > 0.5).astype(np.int32)
    print(f"[v76/train] Stage1: {y_nz.mean()*100:.1f}% non-zero")
    m_cls = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                          n_jobs=-1, random_state=42, verbosity=0)
    m_cls.fit(Xt, y_nz)

    # Stage 2: regressor on non-zero
    mask_nz = np.abs(yt) > 0.5
    Xt_nz = Xt[mask_nz]; yt_nz = yt[mask_nz]
    print(f"[v76/train] Stage2: {len(Xt_nz)} non-zero rows")
    m_reg = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                         n_jobs=-1, random_state=42, verbosity=0)
    m_reg.fit(Xt_nz, yt_nz)

    os.makedirs(model_directory_path, exist_ok=True)
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump((m_cls, m_reg, feature_cols, target_col), f)


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        m_cls, m_reg, feature_cols, _ = pickle.load(f)
    Xt = X_test[feature_cols].values.astype(np.float32)
    p_nonzero = m_cls.predict_proba(Xt)[:, 1]
    p_reg = m_reg.predict(Xt)
    pred = p_nonzero * p_reg
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": pred})
