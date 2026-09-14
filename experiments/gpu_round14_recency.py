"""R14: Recency-weighted training — does emphasizing recent moons help k=9?

Hypothesis: financial regime drifts, so recent moons may be more predictive.
Test 3 decay rates (slow, medium, aggressive) vs uniform baseline.
30 samples = 10 train_ends × 3 seeds.
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
SEEDS = [42, 7, 2026]

def run(decay_name, get_w):
    bl, exp = [], []
    for T in TRAIN_ENDS:
        tm = T + 9
        if tm > merged['moon'].max(): continue
        tr = merged[merged['moon'] <= T]
        moons_tr = tr['moon'].values.astype(np.int32)
        yt = tr['target'].values.astype(np.float32)
        Xt = tr[feat].values.astype(np.float32)
        test = merged[merged['moon'] == tm]
        Xte = test[feat].values.astype(np.float32)
        y_te = test['target'].values
        w = get_w(moons_tr, T).astype(np.float32)
        for s in SEEDS:
            m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                               device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
            m_b.fit(Xt, yt)
            r_b = pearsonr(m_b.predict(Xte), y_te)[0]
            m_w = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                               device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
            m_w.fit(Xt, yt, sample_weight=w)
            r_w = pearsonr(m_w.predict(Xte), y_te)[0]
            bl.append(r_b); exp.append(r_w)
            print(f"  [{decay_name}] T={T:<4} s={s:<5} bl={r_b:+.5f} w={r_w:+.5f} d={r_w-r_b:+.5f}", flush=True)
    bl=np.array(bl); exp=np.array(exp); d=exp-bl
    print(f"\n=== R14 [{decay_name}] (n={len(bl)}) ===")
    print(f"  Baseline: mean={bl.mean():+.5f} std={bl.std():.5f}")
    print(f"  Weighted: mean={exp.mean():+.5f} std={exp.std():.5f}")
    print(f"  Diff:     mean={d.mean():+.5f} std={d.std():.5f}  wins {(d>0).sum()}/{len(d)}  t={d.mean()/(d.std()/np.sqrt(len(d))):.2f}\n")

# slow: half-life 200 moons
print("=== A. Slow decay (half-life=200) ===")
run('slow', lambda m, T: np.power(0.5, (T - m) / 200))
# medium: half-life 100 moons
print("=== B. Medium decay (half-life=100) ===")
run('med',  lambda m, T: np.power(0.5, (T - m) / 100))
# aggressive: only last 200 moons weighted, rest weight 0.2
print("=== C. Last200 emphasis (last200=2, rest=0.2) ===")
run('cut',  lambda m, T: np.where(m >= T-200, 2.0, 0.2))
