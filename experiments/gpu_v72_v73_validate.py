"""Validate v72 (5-model hetero) and v73 (recent-only XGB) with TRUE Pearson-last-moon."""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    TRAIN_ENDS = [500, 550, 600, 650, 670]  # 5 folds for speed
    EMBARGO = 4
    TEST_HORIZON = 5

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from catboost import CatBoostRegressor

    print(f"{'Variant':<35} | {'MeanLast':>9} | {'StdLast':>7} | Per-fold")
    print('-'*120)

    # === v72: 5-model hetero ensemble ===
    v72_scores = []
    for fi, T in enumerate(TRAIN_ENDS):
        t0 = time.time()
        tr = X['moon'] <= T
        te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna(); merged = merged.loc[mask].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        Xte = X_te[feat].values.astype(np.float32)

        m_xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m_xgb.fit(Xt, yt); p_xgb = m_xgb.predict(Xte)
        m_lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                               colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, n_jobs=-1, random_state=42)
        m_lgbm.fit(Xt, yt); p_lgbm = m_lgbm.predict(Xte)
        m_cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=3,
                                 random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1, task_type='GPU', devices='0')
        m_cb.fit(Xt, yt); p_cb = m_cb.predict(Xte)
        imp = SimpleImputer(strategy='median'); Xt_imp = imp.fit_transform(Xt)
        m_ridge = Ridge(alpha=10.0, random_state=42); m_ridge.fit(Xt_imp, yt)
        p_ridge = m_ridge.predict(imp.transform(Xte))
        m_et = ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=200, max_features=0.4,
                                   n_jobs=-1, random_state=42)
        m_et.fit(Xt_imp, yt); p_et = m_et.predict(imp.transform(Xte))

        # Weights based on individual: XGB=3, LGBM=2, CatBoost=1.5, Ridge=0.5, ET=1
        p_ens = (3*p_xgb + 2*p_lgbm + 1.5*p_cb + 0.5*p_ridge + 1*p_et) / 8.0
        pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': p_ens})
        sc = pearson_last_moon(pred, y_te)
        v72_scores.append(sc)
        print(f"  v72 fold {fi+1}/5 T={T} ({time.time()-t0:.0f}s) p={sc:.4f}")
    print(f"v72_5model_hetero {'':<19} | {np.mean(v72_scores):>+9.5f} | {np.std(v72_scores):>7.5f} | {[round(s,4) for s in v72_scores]}")

    # === v73: XGB trained on recent 300 moons only ===
    print("\nv73: train only on recent moons (T-300..T)")
    v73_scores = []
    for fi, T in enumerate(TRAIN_ENDS):
        t0 = time.time()
        T_min = max(1, T - 300)
        tr = (X['moon'] >= T_min) & (X['moon'] <= T)
        te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[(y['moon'] >= T_min) & (y['moon'] <= T)].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna(); merged = merged.loc[mask].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)

        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m.fit(Xt, yt)
        pred_arr = m.predict(X_te[feat].values.astype(np.float32))
        pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': pred_arr})
        sc = pearson_last_moon(pred, y_te)
        v73_scores.append(sc)
        print(f"  v73 fold {fi+1}/5 T={T} ({time.time()-t0:.0f}s) p={sc:.4f}")
    print(f"v73_recent300moons {'':<18} | {np.mean(v73_scores):>+9.5f} | {np.std(v73_scores):>7.5f} | {[round(s,4) for s in v73_scores]}")


if __name__ == '__main__':
    main()
