"""R20: CatBoost vs XGB baseline (30 samples).

CatBoost uses symmetric trees + ordered boosting — different bias/variance vs XGB.
Strong on noisy tabular. Test 2 configs.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
gpu_idx = int(DEV.split(':')[-1])

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]

def run(name, build_cat):
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
            m_a = build_cat(s)
            m_a.fit(Xt, yt, verbose=False)
            r_a = pearsonr(m_a.predict(Xte), y_te)[0]
            bl.append(r_b); alt.append(r_a)
            print(f"  [{name}] T={T:<4} s={s:<5} xgb={r_b:+.5f} cat={r_a:+.5f} diff={r_a-r_b:+.5f}", flush=True)
    bl=np.array(bl); alt=np.array(alt); diff=alt-bl
    print(f"\n=== R20 [{name}] (n={len(bl)}) ===")
    print(f"  XGB baseline: mean={bl.mean():+.5f} std={bl.std():.5f}")
    print(f"  CatBoost {name}: mean={alt.mean():+.5f} std={alt.std():.5f}")
    print(f"  Diff mean={diff.mean():+.5f} std={diff.std():.5f}  wins {(diff>0).sum()}/{len(diff)}  t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}\n")

# A. Standard CatBoost (analogous capacity to XGB baseline)
print("=== A. CatBoost 500/d6/lr=0.03 ===")
run('cat500_d6_lr03', lambda s: CatBoostRegressor(
    iterations=500, depth=6, learning_rate=0.03,
    bootstrap_type='Bernoulli', subsample=0.8,
    task_type='GPU', devices=str(gpu_idx),
    random_seed=s, allow_writing_files=False))

# B. CatBoost with slower lr (mirroring R17 finding)
print("=== B. CatBoost 3000/d6/lr=0.005 ===")
run('cat3000_d6_lr005', lambda s: CatBoostRegressor(
    iterations=3000, depth=6, learning_rate=0.005,
    bootstrap_type='Bernoulli', subsample=0.8,
    task_type='GPU', devices=str(gpu_idx),
    random_seed=s, allow_writing_files=False))
