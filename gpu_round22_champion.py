"""R22: Champion config — combine ALL verified findings into one ensemble.

Verified so far:
- depth=6 optimal (R15)
- slow lr +0.004 (R16-R19)
- hetero XGB+LGBM+Ridge ensemble +0.0094 on cloud (v68)

Test 3 ensembles vs single XGB baseline (30 samples):
  A. Baseline hetero (v68 config): XGB[500/lr.03] + LGBM + Ridge, wts [2,1,0.5]
  B. Slow-lr hetero: XGB[3000/lr.005] + LGBM[3000/lr.005] + Ridge, wts [2,1,0.5]
  C. Slow-lr hetero 3-seed avg of B
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
gpu_idx = int(DEV.split(':')[-1])

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]

def zscore_pred(p):
    return (p - p.mean()) / (p.std() + 1e-9)

def hetero(Xt, yt, Xte, s, slow=False):
    if slow:
        xgb = XGBRegressor(n_estimators=3000, max_depth=6, learning_rate=0.005,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device=DEV, n_jobs=-1, random_state=s, verbosity=0)
        lgb = LGBMRegressor(n_estimators=3000, num_leaves=63, max_depth=6, learning_rate=0.005,
                            subsample=0.8, colsample_bytree=0.5, subsample_freq=1,
                            device='gpu', gpu_device_id=gpu_idx, random_state=s, verbosity=-1, n_jobs=-1)
    else:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device=DEV, n_jobs=-1, random_state=s, verbosity=0)
        lgb = LGBMRegressor(n_estimators=500, num_leaves=63, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.5, subsample_freq=1,
                            device='gpu', gpu_device_id=gpu_idx, random_state=s, verbosity=-1, n_jobs=-1)
    xgb.fit(Xt, yt); lgb.fit(Xt, yt)
    sc = StandardScaler().fit(Xt)
    rdg = Ridge(alpha=10.0, random_state=s).fit(sc.transform(Xt), yt)
    px = zscore_pred(xgb.predict(Xte))
    pl = zscore_pred(lgb.predict(Xte))
    pr = zscore_pred(rdg.predict(sc.transform(Xte)))
    return (2*px + 1*pl + 0.5*pr) / 3.5

def run(name, predict_fn):
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
            r_a = pearsonr(predict_fn(Xt, yt, Xte, s), y_te)[0]
            bl.append(r_b); alt.append(r_a)
            print(f"  [{name}] T={T:<4} s={s:<5} xgb={r_b:+.5f} ens={r_a:+.5f} diff={r_a-r_b:+.5f}", flush=True)
    bl=np.array(bl); alt=np.array(alt); diff=alt-bl
    print(f"\n=== R22 [{name}] (n={len(bl)}) ===")
    print(f"  XGB single baseline: mean={bl.mean():+.5f} std={bl.std():.5f}")
    print(f"  {name}:              mean={alt.mean():+.5f} std={alt.std():.5f}")
    print(f"  Diff mean={diff.mean():+.5f} std={diff.std():.5f}  wins {(diff>0).sum()}/{len(diff)}  t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}\n")

print("=== A. Baseline hetero (v68 fast lr) ===")
run('hetero_fast', lambda Xt,yt,Xte,s: hetero(Xt,yt,Xte,s,slow=False))

print("=== B. Slow-lr hetero ===")
run('hetero_slow', lambda Xt,yt,Xte,s: hetero(Xt,yt,Xte,s,slow=True))
