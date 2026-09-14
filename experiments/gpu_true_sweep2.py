"""Round 2 of TRUE validator: explore around v14 winner.

v14 = 500/d6/lr=0.03 RAW target → 0.0648 MeanLast
This sweep tests close neighbors + tweaks that A5 recommended.
"""
import pandas as pd, numpy as np, time, json, sys
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


# Around v14 winner: 500/d6/lr=0.03 → variants
CONFIGS = [
    # v14 reference
    ('500/d6/lr=0.03 (v14 anchor)',  dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=100)),
    # Small neighborhood
    ('400/d6/lr=0.03',               dict(n_estimators=400, max_depth=6, learning_rate=0.03, min_child_weight=100)),
    ('600/d6/lr=0.03',               dict(n_estimators=600, max_depth=6, learning_rate=0.03, min_child_weight=100)),
    ('500/d6/lr=0.025',              dict(n_estimators=500, max_depth=6, learning_rate=0.025, min_child_weight=100)),
    ('500/d6/lr=0.04',               dict(n_estimators=500, max_depth=6, learning_rate=0.04, min_child_weight=100)),
    # Higher min_child_weight (A5 said >=500)
    ('500/d6/lr=0.03 mcw=500',       dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=500)),
    ('500/d6/lr=0.03 mcw=1000',      dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=1000)),
    # Different col/sub
    ('500/d6/lr=0.03 col=0.7',       dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=100, colsample_bytree=0.7)),
    ('500/d6/lr=0.03 sub=1.0',       dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=100, subsample=1.0)),
    # Try moderate d=7 with regularization
    ('500/d7/lr=0.03 mcw=500',       dict(n_estimators=500, max_depth=7, learning_rate=0.03, min_child_weight=500)),
]

TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
EMBARGO = 4
TEST_HORIZON = 5


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    splits = []
    for T in TRAIN_ENDS:
        tr = X['moon'] <= T
        te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        if X_te.empty: continue
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'Xt': Xt, 'yt': yt, 'X_te': X_te, 'y_te': y_te})

    from xgboost import XGBRegressor
    print(f"{'Config':<35} | {'MeanLast':>9} | {'StdLast':>7} | Per-fold last-moon")
    print('-'*120)
    results = []
    for name, cfg in CONFIGS:
        last_scores = []
        t0 = time.time()
        for s in splits:
            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0)
            params.update(cfg)
            m = XGBRegressor(**params)
            m.fit(s['Xt'], s['yt'])
            pred_arr = m.predict(s['X_te'][feat].values.astype(np.float32))
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            ls = pearson_last_moon(pred, s['y_te'])
            last_scores.append(ls)
        mean_s = float(np.mean(last_scores))
        std_s = float(np.std(last_scores))
        elapsed = time.time() - t0
        results.append({'name': name, **cfg, 'mean_last_moon': mean_s, 'std_last_moon': std_s, 'folds': last_scores, 'time': elapsed})
        print(f"{name:<35} | {mean_s:>+9.5f} | {std_s:>7.5f} | {[round(s, 4) for s in last_scores]} ({elapsed:.0f}s)")
        with open('/workspace/sweep_true2.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
