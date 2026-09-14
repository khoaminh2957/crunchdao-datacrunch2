"""R11: Verify sw=2 lift with 30 samples (10 train_ends × 3 seeds)."""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]

bl_all, sw_all = [], []
for T in TRAIN_ENDS:
    tm = T + 9
    if tm > merged['moon'].max(): continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test = merged[merged['moon'] == tm]
    Xte = test[feat].values.astype(np.float32)
    y_te = test['target'].values
    w = np.where(np.abs(yt) > 0.5, 2.0, 1.0).astype(np.float32)
    for s in SEEDS:
        # Baseline
        m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
        m_b.fit(Xt, yt)
        r_b = pearsonr(m_b.predict(Xte), y_te)[0]
        # sw=2
        m_w = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
        m_w.fit(Xt, yt, sample_weight=w)
        r_w = pearsonr(m_w.predict(Xte), y_te)[0]
        bl_all.append(r_b); sw_all.append(r_w)
        print(f"T={T:<5} seed={s:<5} bl={r_b:+.5f} sw2={r_w:+.5f} diff={r_w-r_b:+.5f}", flush=True)

bl_all = np.array(bl_all); sw_all = np.array(sw_all); diffs = sw_all - bl_all
print(f"\n=== R11 sw=2 verify (n={len(bl_all)}) ===")
print(f"  Baseline: mean={bl_all.mean():+.5f} std={bl_all.std():.5f}")
print(f"  sw=2:     mean={sw_all.mean():+.5f} std={sw_all.std():.5f}")
print(f"  Diff:     mean={diffs.mean():+.5f} std={diffs.std():.5f}")
print(f"  sw=2 wins: {(diffs>0).sum()}/{len(diffs)} = {(diffs>0).mean()*100:.0f}%")
print(f"  t-stat: {diffs.mean()/(diffs.std()/np.sqrt(len(diffs))):.2f}")
