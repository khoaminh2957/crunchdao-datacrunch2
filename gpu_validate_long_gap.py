"""Validate top configs with 100-moon gap (mimics cloud test setup).

Train ends T, test on T+100..T+109 (100-moon gap matches cloud train→test distance).
This should better predict cloud score.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr


def rank_per_moon(y_df):
    grp = y_df.groupby('moon')['target']
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def per_moon_spearman(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    scores = []
    for m, sub in merged.groupby('moon'):
        if len(sub) < 5: continue
        r = spearmanr(sub['prediction'], sub['target'])[0]
        if np.isfinite(r): scores.append(r)
    return float(np.mean(scores)) if scores else float('nan')


# Configs to validate — RUN VARIETY of model capacities
CONFIGS = [
    ('500/d6/lr=0.03 (v14)',     dict(n_estimators=500, max_depth=6, learning_rate=0.03)),
    ('500/d5/lr=0.05',           dict(n_estimators=500, max_depth=5, learning_rate=0.05)),
    ('300/d5/lr=0.05',           dict(n_estimators=300, max_depth=5, learning_rate=0.05)),
    ('700/d6/lr=0.05',           dict(n_estimators=700, max_depth=6, learning_rate=0.05)),
    ('700/d7/lr=0.05 (v50)',     dict(n_estimators=700, max_depth=7, learning_rate=0.05)),
    ('1500/d8/lr=0.04 (v52)',    dict(n_estimators=1500, max_depth=8, learning_rate=0.04)),
    ('1500/d9/lr=0.07 (v51)',    dict(n_estimators=1500, max_depth=9, learning_rate=0.07)),
]

# Folds with 100-moon gap (simulate cloud setup)
# Train 1..T, test T+91..T+99 (90-100 moon gap)
TRAIN_ENDS = [550, 600, 650]  # 3 folds (smaller because less data)
TEST_OFFSET = 90  # gap of 90 moons
TEST_HORIZON = 9


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    splits = []
    for T in TRAIN_ENDS:
        tr = X['moon'] <= T
        te = (X['moon'] >= T + TEST_OFFSET) & (X['moon'] < T + TEST_OFFSET + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] >= T + TEST_OFFSET) & (y['moon'] < T + TEST_OFFSET + TEST_HORIZON)].reset_index(drop=True)
        if X_te.empty: continue
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_ranked = rank_per_moon(merged[['moon','target']]).values
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'test_range': (T+TEST_OFFSET, T+TEST_OFFSET+TEST_HORIZON-1),
                       'Xt': Xt, 'yt': y_ranked, 'X_te': X_te, 'y_te': y_te})
        print(f"  T={T} → test moons {T+TEST_OFFSET}-{T+TEST_OFFSET+TEST_HORIZON-1}: train={Xt.shape}, test={X_te.shape}")

    from xgboost import XGBRegressor
    print(f"\n{'Config':<30} | {'Mean':>10} | {'Std':>8} | Per-fold scores")
    print('-'*100)
    results = []
    for name, cfg in CONFIGS:
        fold_scores = []
        for s in splits:
            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
                          subsample=0.8, colsample_bytree=0.5, min_child_weight=1.0, reg_lambda=1.0)
            params.update(cfg)
            m = XGBRegressor(**params)
            m.fit(s['Xt'], s['yt'])
            pred_arr = m.predict(s['X_te'][feat].values.astype(np.float32))
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            sc = per_moon_spearman(pred, s['y_te'])
            fold_scores.append(sc)
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        results.append({'name': name, 'mean': mean_s, 'std': std_s, 'folds': fold_scores})
        print(f"{name:<30} | {mean_s:>+10.5f} | {std_s:>8.5f} | {[round(s, 4) for s in fold_scores]}")


if __name__ == '__main__':
    main()
