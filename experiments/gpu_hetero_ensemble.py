"""B2: Heterogeneous ensemble (XGB + LGBM + Ridge) — different families."""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')


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
    print(f"X={X.shape}")

    TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
    EMBARGO = 4
    TEST_HORIZON = 5

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer

    xgb_scores, lgbm_scores, ridge_scores, ens_scores = [], [], [], []
    t_global = time.time()
    for fi, T in enumerate(TRAIN_ENDS):
        t0 = time.time()
        tr = X['moon'] <= T
        te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        X_te_arr = X_te[feat].values.astype(np.float32)

        # XGB
        m_xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.5,
                             tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m_xgb.fit(Xt, yt)
        p_xgb = m_xgb.predict(X_te_arr)

        # LGBM
        m_lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                               min_data_in_leaf=200, colsample_bytree=0.5, reg_lambda=2.0,
                               verbose=-1, n_jobs=-1, random_state=42)
        m_lgbm.fit(Xt, yt)
        p_lgbm = m_lgbm.predict(X_te_arr)

        # Ridge with imputer
        imp = SimpleImputer(strategy='median')
        Xt_imp = imp.fit_transform(Xt)
        Xte_imp = imp.transform(X_te_arr)
        m_ridge = Ridge(alpha=10.0, random_state=42)
        m_ridge.fit(Xt_imp, yt)
        p_ridge = m_ridge.predict(Xte_imp)

        # Scores
        for name, p_arr, scores in [('xgb', p_xgb, xgb_scores), ('lgbm', p_lgbm, lgbm_scores), ('ridge', p_ridge, ridge_scores)]:
            pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': p_arr})
            scores.append(pearson_last_moon(pred, y_te))

        # Weighted ensemble: XGB=2, LGBM=1, Ridge=0.5
        p_ens = (2 * p_xgb + 1 * p_lgbm + 0.5 * p_ridge) / 3.5
        pred_ens = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': p_ens})
        ens_scores.append(pearson_last_moon(pred_ens, y_te))

        print(f"  Fold {fi+1}/10 T={T} ({time.time()-t0:.0f}s): xgb={xgb_scores[-1]:.4f} lgbm={lgbm_scores[-1]:.4f} ridge={ridge_scores[-1]:.4f} ens={ens_scores[-1]:.4f}")

    print(f"\n=== Final ({time.time()-t_global:.0f}s total) ===")
    for name, s in [('XGB', xgb_scores), ('LGBM', lgbm_scores), ('Ridge', ridge_scores), ('Weighted Ensemble', ens_scores)]:
        print(f"  {name:<20} mean={np.mean(s):+.5f}  std={np.std(s):.5f}  folds={[round(x,4) for x in s]}")


if __name__ == '__main__':
    main()
