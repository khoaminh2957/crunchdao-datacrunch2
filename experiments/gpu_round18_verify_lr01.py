"""R18: 60-sample verification of lr=0.01 n=1500 — does the trend hold?

R16: lr=0.01 n=1500 vs lr=0.03 n=500: diff +0.00512, t=1.54 (p~0.13, 30 samples).
60 samples = 10 train_ends × 6 seeds. Goal: get t-stat above 2 (p<0.05) OR
falsify like all the others.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026, 1337, 99, 333]  # 6 seeds

bl, alt = [], []
for T in TRAIN_ENDS:
    tm = T + 9
    if tm > merged['moon'].max(): continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test = merged[merged['moon'] == tm]
    Xte = test[feat].values.astype(np.float32)
    y_te = test['target'].values
    for s in SEEDS:
        m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device=DEV, n_jobs=-1, random_state=s, verbosity=0)
        m_b.fit(Xt, yt)
        r_b = pearsonr(m_b.predict(Xte), y_te)[0]
        m_a = XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.01,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device=DEV, n_jobs=-1, random_state=s, verbosity=0)
        m_a.fit(Xt, yt)
        r_a = pearsonr(m_a.predict(Xte), y_te)[0]
        bl.append(r_b); alt.append(r_a)
        print(f"  T={T:<4} s={s:<5} bl={r_b:+.5f} lr01={r_a:+.5f} diff={r_a-r_b:+.5f}", flush=True)

bl=np.array(bl); alt=np.array(alt); diff=alt-bl
print(f"\n=== R18 lr=0.01 n=1500 verify (n={len(bl)}) ===")
print(f"  Baseline lr=0.03 n=500: mean={bl.mean():+.5f} std={bl.std():.5f}")
print(f"  Alt lr=0.01 n=1500:     mean={alt.mean():+.5f} std={alt.std():.5f}")
print(f"  Diff mean={diff.mean():+.5f} std={diff.std():.5f}  wins {(diff>0).sum()}/{len(diff)}  t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}")
