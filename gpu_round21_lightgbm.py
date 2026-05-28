"""R21: LightGBM vs XGB baseline (30 samples).

LightGBM uses leaf-wise (vs XGB level-wise) growth + histogram + different reg.
Test 3 configs: standard, slow-lr (mirroring R17), num_leaves variant.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
gpu_idx = int(DEV.split(':')[-1])

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]

def run(name, build_lgbm):
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
            m_a = build_lgbm(s)
            m_a.fit(Xt, yt)
            r_a = pearsonr(m_a.predict(Xte), y_te)[0]
            bl.append(r_b); alt.append(r_a)
            print(f"  [{name}] T={T:<4} s={s:<5} xgb={r_b:+.5f} lgbm={r_a:+.5f} diff={r_a-r_b:+.5f}", flush=True)
    bl=np.array(bl); alt=np.array(alt); diff=alt-bl
    print(f"\n=== R21 [{name}] (n={len(bl)}) ===")
    print(f"  XGB baseline:  mean={bl.mean():+.5f} std={bl.std():.5f}")
    print(f"  LightGBM {name}: mean={alt.mean():+.5f} std={alt.std():.5f}")
    print(f"  Diff mean={diff.mean():+.5f} std={diff.std():.5f}  wins {(diff>0).sum()}/{len(diff)}  t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}\n")

# A. LightGBM standard (matching XGB baseline capacity)
print("=== A. LGBM 500/leaves63/lr=0.03 ===")
run('lgbm_std', lambda s: LGBMRegressor(
    n_estimators=500, num_leaves=63, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.5, subsample_freq=1,
    device='gpu', gpu_device_id=gpu_idx,
    random_state=s, verbosity=-1, n_jobs=-1))

# B. LightGBM slow lr (mirror R17 winner)
print("=== B. LGBM 3000/leaves63/lr=0.005 ===")
run('lgbm_slow', lambda s: LGBMRegressor(
    n_estimators=3000, num_leaves=63, max_depth=6, learning_rate=0.005,
    subsample=0.8, colsample_bytree=0.5, subsample_freq=1,
    device='gpu', gpu_device_id=gpu_idx,
    random_state=s, verbosity=-1, n_jobs=-1))

# C. LightGBM more leaves (leaf-wise advantage)
print("=== C. LGBM 500/leaves127/lr=0.03 ===")
run('lgbm_wide', lambda s: LGBMRegressor(
    n_estimators=500, num_leaves=127, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.5, subsample_freq=1,
    device='gpu', gpu_device_id=gpu_idx,
    random_state=s, verbosity=-1, n_jobs=-1))
