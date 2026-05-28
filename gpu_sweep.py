"""Hyperparameter sweep on GPU server — find best XGB config via walk-forward CV.

Strategy:
- ~40 configs spanning n_est × depth × lr × colsample × subsample × min_child × reg_lambda
- 5-fold walk-forward CV per config (matches cloud test setup)
- Rank by mean Spearman, break ties by lower std (stability)
- Save top 5 configs to submit_top5.json

Estimated cost on 2x RTX 5060 Ti:
- Each XGB ~30-60s on GPU
- 5 folds × 40 configs = 200 XGB trains = ~2-3 hours

Run: python gpu_sweep.py
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


# Sweep grid (40 configs)
SWEEP = []
for n_est in [300, 500, 700]:
    for depth in [5, 6, 7]:
        for lr in [0.02, 0.03, 0.05]:
            SWEEP.append({'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
                          'subsample': 0.8, 'colsample_bytree': 0.5,
                          'min_child_weight': 1.0, 'reg_lambda': 1.0})
# Add some specials
SWEEP += [
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 1.0, 'colsample_bytree': 0.5, 'min_child_weight': 1.0, 'reg_lambda': 1.0},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.6, 'colsample_bytree': 0.4, 'min_child_weight': 1.0, 'reg_lambda': 1.0},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 10.0, 'reg_lambda': 5.0},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 1.0, 'reg_lambda': 1.0},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.3, 'min_child_weight': 1.0, 'reg_lambda': 1.0},
]
print(f"Total configs: {len(SWEEP)}")

# Walk-forward train ends + test horizon = 9 (matches cloud)
TRAIN_ENDS = [610, 625, 640, 655, 670]
TEST_HORIZON = 9


def main():
    print("Loading data...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}, y={y.shape}, feats={len(feat)}")

    # Pre-compute per-fold splits to avoid repeated boolean masks
    splits = []
    for T in TRAIN_ENDS:
        tr_mask = X['moon'] <= T
        te_mask = (X['moon'] > T) & (X['moon'] <= T + TEST_HORIZON)
        X_tr = X[tr_mask].reset_index(drop=True)
        X_te = X[te_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T) & (y['moon'] <= T + TEST_HORIZON)].reset_index(drop=True)
        # Pre-merge train
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_ranked = rank_per_moon(merged[['moon','target']]).values
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'Xt': Xt, 'yt': y_ranked, 'X_te': X_te, 'y_te': y_te})
        print(f"  Split T={T}: train={Xt.shape}, test={X_te.shape}, test_moons={y_te['moon'].nunique()}")

    from xgboost import XGBRegressor

    results = []
    for cfg_i, cfg in enumerate(SWEEP):
        fold_scores = []
        t_start = time.time()
        for s in splits:
            params = dict(tree_method='hist', device='cuda', n_jobs=-1,
                         random_state=42, verbosity=0)
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
              f"sub={cfg['subsample']} col={cfg['colsample_bytree']} mcw={cfg['min_child_weight']} reg={cfg['reg_lambda']}")
        # Save incremental
        with open('/workspace/sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Rank
    df = pd.DataFrame(results).sort_values(['mean_spearman','std_spearman'], ascending=[False, True])
    print("\n=== TOP 10 ===")
    print(df.head(10).to_string(index=False))
    df.to_csv('/workspace/sweep_results.csv', index=False)
    top5 = df.head(5).to_dict('records')
    with open('/workspace/submit_top5.json', 'w') as f:
        json.dump(top5, f, indent=2)
    print(f"\nSaved top 5 to /workspace/submit_top5.json")


if __name__ == '__main__':
    main()
