"""R13: Verify if 5-seed XGB ensemble beats single seed (30 samples).

Hypothesis: avg of 5 seeds reduces variance and may lift mean correlation.
This is the most defensible 'lift' candidate since variance reduction is
mathematically guaranteed for uncorrelated noise components.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS_SINGLE = [42, 7, 2026]
SEEDS_ENS = [42, 7, 2026, 1337, 99]

bl_all, ens_all = [], []
for T in TRAIN_ENDS:
    tm = T + 9
    if tm > merged['moon'].max(): continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test = merged[merged['moon'] == tm]
    Xte = test[feat].values.astype(np.float32)
    y_te = test['target'].values
    # Single seeds (baseline)
    for s in SEEDS_SINGLE:
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                         device='cuda:1', n_jobs=-1, random_state=s, verbosity=0)
        m.fit(Xt, yt)
        r = pearsonr(m.predict(Xte), y_te)[0]
        bl_all.append(r)
        # 5-seed ensemble (using same s as base seed for fairness)
        preds = []
        for es in SEEDS_ENS:
            me = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                              device='cuda:1', n_jobs=-1, random_state=es*1000+s, verbosity=0)
            me.fit(Xt, yt)
            preds.append(me.predict(Xte))
        ens = np.mean(preds, axis=0)
        r_e = pearsonr(ens, y_te)[0]
        ens_all.append(r_e)
        print(f"T={T:<5} base_seed={s:<5} single={r:+.5f} ens5={r_e:+.5f} diff={r_e-r:+.5f}", flush=True)

bl_all = np.array(bl_all); ens_all = np.array(ens_all); diffs = ens_all - bl_all
print(f"\n=== R13 5-seed ensemble verify (n={len(bl_all)}) ===")
print(f"  Single seed:  mean={bl_all.mean():+.5f} std={bl_all.std():.5f}")
print(f"  5-seed ens:   mean={ens_all.mean():+.5f} std={ens_all.std():.5f}")
print(f"  Diff:         mean={diffs.mean():+.5f} std={diffs.std():.5f}")
print(f"  Ens wins: {(diffs>0).sum()}/{len(diffs)} = {(diffs>0).mean()*100:.0f}%")
print(f"  t-stat: {diffs.mean()/(diffs.std()/np.sqrt(len(diffs))):.2f}")
print(f"  Var reduction: {bl_all.std()/ens_all.std():.2f}x")
