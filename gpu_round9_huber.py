"""Round 9: verify Huber loss lift with more samples.

R8 claim: Huber +0.005 mean, 2.6x less variance vs baseline (5 samples).
R9: 10 train_ends × 3 seeds = 30 samples for tight CI.
"""
import pandas as pd, numpy as np, time, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]
HORIZON_K = 9

bl_all = []
hb_all = []
diffs = []
for T in TRAIN_ENDS:
    tm = T + HORIZON_K
    if tm > merged['moon'].max(): continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test = merged[merged['moon'] == tm]
    Xte = test[feat].values.astype(np.float32)
    y_te = test['target'].values
    for s in SEEDS:
        # Baseline
        m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:1', n_jobs=-1, random_state=s, verbosity=0)
        m_b.fit(Xt, yt)
        r_b = pearsonr(m_b.predict(Xte), y_te)[0]
        # Huber
        m_h = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           objective='reg:pseudohubererror',
                           device='cuda:1', n_jobs=-1, random_state=s, verbosity=0)
        m_h.fit(Xt, yt)
        r_h = pearsonr(m_h.predict(Xte), y_te)[0]
        bl_all.append(r_b); hb_all.append(r_h); diffs.append(r_h - r_b)
        print(f"T={T:<5} seed={s:<5} bl={r_b:+.5f} huber={r_h:+.5f} diff={r_h-r_b:+.5f}", flush=True)

bl_all = np.array(bl_all); hb_all = np.array(hb_all); diffs = np.array(diffs)
print(f"\n=== R9 RESULTS (n={len(bl_all)}) ===")
print(f"  Baseline: mean={bl_all.mean():+.5f} std={bl_all.std():.5f}")
print(f"  Huber:    mean={hb_all.mean():+.5f} std={hb_all.std():.5f}")
print(f"  Lift mean: {diffs.mean():+.5f}  std: {diffs.std():.5f}")
print(f"  Paired t: {diffs.mean()/(diffs.std()/np.sqrt(len(diffs))):.2f}")
print(f"  Huber wins: {(diffs > 0).sum()}/{len(diffs)} = {(diffs > 0).mean()*100:.1f}%")
print(f"\n  Variance ratio (bl_std / hb_std): {bl_all.std()/hb_all.std():.2f}x")
