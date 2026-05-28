"""Horizon-aware validator: test SPECIFICALLY on the moon-k-after-train, like cloud does.

Mimics cloud: train 1-T, score on moon (T+9) only. Multiple T values.
"""
import pandas as pd, numpy as np, time, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

# Test specifically k=9 horizon (matches cloud last moon)
TRAIN_ENDS = [550, 600, 650, 700, 750, 760, 765, 770, 772]
HORIZON_K = 9

from xgboost import XGBRegressor

# 5 seeds per train end for variance
SEEDS = [42, 7, 2026, 11, 99]

print(f"Horizon=k=9 validator. 9 train_ends × 5 seeds = 45 trains.")
print(f"{'TrainEnd':<8} | seeds Pearson (k=9) | mean | std")
print('-'*100)
all_results = []
for T in TRAIN_ENDS:
    test_m = T + HORIZON_K
    if test_m > merged['moon'].max():
        continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test_sub = merged[merged['moon'] == test_m]
    Xte = test_sub[feat].values.astype(np.float32)
    y_te = test_sub['target'].values

    seed_scores = []
    for s in SEEDS:
        t0 = time.time()
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.5, tree_method='hist', device='cuda:1', n_jobs=-1, random_state=s, verbosity=0)
        m.fit(Xt, yt)
        p = m.predict(Xte)
        r = pearsonr(p, y_te)[0]
        seed_scores.append(r)
    all_results.append({'T': T, 'test_moon': test_m, 'scores': seed_scores, 'mean': np.mean(seed_scores), 'std': np.std(seed_scores)})
    print(f"T={T:<5} → moon {test_m}: {[f'{s:+.4f}' for s in seed_scores]} mean={np.mean(seed_scores):+.4f} std={np.std(seed_scores):.4f}")

# Overall mean (this is the validator MEAN for k=9 = cloud expectation)
all_means = [r['mean'] for r in all_results]
all_scores_flat = [s for r in all_results for s in r['scores']]
print(f"\n=== Overall k=9 stats (validator prediction for cloud) ===")
print(f"  Across {len(TRAIN_ENDS)} train-ends × 5 seeds = {len(all_scores_flat)} samples")
print(f"  Mean: {np.mean(all_scores_flat):+.5f}")
print(f"  Std:  {np.std(all_scores_flat):.5f}")
print(f"  Within-T std: {np.mean([r['std'] for r in all_results]):.5f}  (seed variance)")
print(f"  Across-T std: {np.std(all_means):.5f}  (data variance)")
print(f"\n  Compare to our 13 cloud single-seed XGB samples: mean=0.053, std=0.024")

with open('/workspace/results/horizon_validator.json', 'w') as f:
    json.dump(all_results, f, indent=2)
