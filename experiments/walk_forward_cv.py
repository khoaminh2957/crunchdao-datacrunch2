"""Walk-forward CV harness — proper validation matching cloud test setup.

Folds: train ends at moon T, test on T+1..T+9 (9 moons like cloud).
Default 5 folds: T ∈ {610, 625, 640, 655, 670}; trains on 1..T, tests on T+1..T+9.
Mean & std across folds give a more reliable estimate than single holdout.
"""
import pandas as pd
import numpy as np
import sys
import time
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')


def rank_per_moon(y_df, moon_col='moon', target_col='target'):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def per_moon_spearman(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    scores = []
    for m, sub in merged.groupby('moon'):
        if len(sub) < 5: continue
        r = spearmanr(sub['prediction'], sub['target'])[0]
        if np.isfinite(r):
            scores.append(r)
    return np.mean(scores), len(scores)


def walk_forward(X, y, feature_cols, train_fn, infer_fn, train_ends=(610, 625, 640, 655, 670), test_horizon=9):
    """Run walk-forward CV; return per-fold scores."""
    print(f"Walk-forward CV: {len(train_ends)} folds, test_horizon={test_horizon}")
    results = []
    for T in train_ends:
        train_mask = X['moon'] <= T
        test_mask = (X['moon'] > T) & (X['moon'] <= T + test_horizon)
        X_tr = X[train_mask].reset_index(drop=True)
        X_te = X[test_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T) & (y['moon'] <= T + test_horizon)].reset_index(drop=True)

        t0 = time.time()
        model = train_fn(X_tr, y_tr, feature_cols)
        train_t = time.time() - t0
        t1 = time.time()
        pred = infer_fn(model, X_te, feature_cols)
        infer_t = time.time() - t1
        score, n = per_moon_spearman(pred, y_te)
        results.append({'T': T, 'score': score, 'n_test_moons': n, 'train_s': train_t, 'infer_s': infer_t})
        print(f"  T={T}: spearman={score:+.5f} (n={n} moons), train={train_t:.1f}s, infer={infer_t:.1f}s")

    df = pd.DataFrame(results)
    print(f"\n  MEAN  : {df['score'].mean():+.5f}")
    print(f"  STD   : {df['score'].std():.5f}")
    print(f"  MEDIAN: {df['score'].median():+.5f}")
    return df


def xgb_train_fn(use_gpu=False, **kwargs):
    """Factory: returns train_fn closure."""
    def f(X_tr, y_tr, feature_cols):
        from xgboost import XGBRegressor
        merged = X_tr[['id','moon']+feature_cols].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_ranked = rank_per_moon(merged[['moon','target']]).values
        Xt = merged[feature_cols]
        params = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.5,
                      tree_method='hist', n_jobs=-1, random_state=42, verbosity=0)
        if use_gpu:
            params['device'] = 'cuda'
        params.update(kwargs)
        m = XGBRegressor(**params)
        m.fit(Xt, y_ranked)
        return m
    return f


def xgb_infer_fn():
    def f(model, X_te, feature_cols):
        preds = model.predict(X_te[feature_cols])
        return pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': preds})
    return f


if __name__ == '__main__':
    print("Loading data...")
    X = pd.read_parquet("C:/Users/Admin/Downloads/X.reduced.parquet")
    y = pd.read_parquet("C:/Users/Admin/Downloads/y.reduced.parquet")
    print(f"X={X.shape}, y={y.shape}")

    feat_v14 = [c for c in X.columns if c not in ('id','moon')]
    print(f"\n=== V14 baseline (n_est=500) on 5-fold walk-forward (CPU) ===")
    df = walk_forward(X, y, feat_v14, xgb_train_fn(use_gpu=False), xgb_infer_fn())
    df.to_csv("wfcv_v14_cpu.csv", index=False)
