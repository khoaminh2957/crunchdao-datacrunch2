"""Round 2 sweep: extend the top region (depth=7/lr=0.05 winner).

Test deeper trees + more trees + higher lr to find if we're plateaued.
"""
import pandas as pd
import numpy as np
import json
import time
import sys
import os
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
        if np.isfinite(r):
            scores.append(r)
    return float(np.mean(scores)) if scores else float('nan')


SWEEP = []
# Extend top region: d=8/9, n=1000/1500, lr=0.05/0.07/0.1
for n_est in [700, 1000, 1500]:
    for depth in [7, 8, 9]:
        for lr in [0.04, 0.05, 0.07]:
            SWEEP.append({'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
                          'subsample': 0.8, 'colsample_bytree': 0.5,
                          'min_child_weight': 1.0, 'reg_lambda': 1.0})
# Add 2 specials at the winning config family
SWEEP += [
    {'n_estimators': 700, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.6, 'min_child_weight': 1.0, 'reg_lambda': 1.0},
    {'n_estimators': 700, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 1.0, 'reg_lambda': 0.5},
]
print(f"Total configs: {len(SWEEP)}")

TRAIN_ENDS = [610, 625, 640, 655, 670]
TEST_HORIZON = 9


def main():
    print("Loading data...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}, y={y.shape}, feats={len(feat)}")

    splits = []
    for T in TRAIN_ENDS:
        tr_mask = X['moon'] <= T
        te_mask = (X['moon'] > T) & (X['moon'] <= T + TEST_HORIZON)
        X_tr = X[tr_mask].reset_index(drop=True)
        X_te = X[te_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T) & (y['moon'] <= T + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_ranked = rank_per_moon(merged[['moon','target']]).values
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'Xt': Xt, 'yt': y_ranked, 'X_te': X_te, 'y_te': y_te})
        print(f"  Split T={T}: train={Xt.shape}, test={X_te.shape}")

    from xgboost import XGBRegressor

    results = []
    for cfg_i, cfg in enumerate(SWEEP):
        fold_scores = []
        t_start = time.time()
        for s in splits:
            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
            params.update(cfg)
            model = XGBRegressor(**params)
            model.fit(s['Xt'], s['yt'])
            pred_arr = model.predict(s['X_te'][feat].values.astype(np.float32))
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            score = per_moon_spearman(pred, s['y_te'])
            fold_scores.append(score)
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        elapsed = time.time() - t_start
        results.append({**cfg, 'mean_spearman': mean_s, 'std_spearman': std_s, 'fold_scores': fold_scores, 'elapsed_s': elapsed})
        print(f"[{cfg_i+1}/{len(SWEEP)}] mean={mean_s:+.5f} std={std_s:.5f} t={elapsed:.1f}s | "
              f"n={cfg['n_estimators']} d={cfg['max_depth']} lr={cfg['learning_rate']:.3f} "
              f"sub={cfg['subsample']} col={cfg['colsample_bytree']}")
        with open('/workspace/sweep_results_r2.json', 'w') as f:
            json.dump(results, f, indent=2)

    df = pd.DataFrame(results).sort_values(['mean_spearman','std_spearman'], ascending=[False, True])
    print("\n=== TOP 10 (round 2) ===")
    print(df.head(10).to_string(index=False))
    df.to_csv('/workspace/sweep_results_r2.csv', index=False)


if __name__ == '__main__':
    main()
