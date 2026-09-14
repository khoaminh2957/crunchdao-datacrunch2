"""R17: even slower lr — extend the R16 gradient.

R16 showed lr=0.01 n=1500 was best (+0.0051 vs base lr=0.03 n=500).
Test if lr=0.005 n=3000 (equivalent capacity) is even better.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]

def run(name, lr, n_est):
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
            m_a = XGBRegressor(n_estimators=n_est, max_depth=6, learning_rate=lr,
                               subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                               device=DEV, n_jobs=-1, random_state=s, verbosity=0)
            m_a.fit(Xt, yt)
            r_a = pearsonr(m_a.predict(Xte), y_te)[0]
            bl.append(r_b); alt.append(r_a)
            print(f"  [{name}] T={T:<4} s={s:<5} bl={r_b:+.5f} alt={r_a:+.5f} diff={r_a-r_b:+.5f}", flush=True)
    bl=np.array(bl); alt=np.array(alt); diff=alt-bl
    print(f"\n=== R17 [{name}] (n={len(bl)}) ===")
    print(f"  Baseline lr=0.03 n=500: mean={bl.mean():+.5f} std={bl.std():.5f}")
    print(f"  Alt {name}:             mean={alt.mean():+.5f} std={alt.std():.5f}")
    print(f"  Diff mean={diff.mean():+.5f} std={diff.std():.5f}  wins {(diff>0).sum()}/{len(diff)}  t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}\n")

print("=== A. lr=0.005 n=3000 ==="); run('lr005_n3000', 0.005, 3000)
print("=== B. lr=0.003 n=5000 ==="); run('lr003_n5000', 0.003, 5000)
