"""Train 10-seed XGB ensemble LOCALLY on full data. Save to model.pkl. Ship via curly resources/.

main.py will load + infer only. Set Force first train=No on trigger.
"""
import pandas as pd, numpy as np, pickle, os, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
print(f"X={X.shape}, y={y.shape}, features={len(feat)}")

merged = X[['id','moon']+feat].merge(y, on=['id','moon'], how='inner')
mask = merged['target'].notna()
merged = merged.loc[mask].reset_index(drop=True)
yt = merged['target'].values.astype(np.float32)
Xt = merged[feat].values.astype(np.float32)
print(f"Training data: {Xt.shape}")

from xgboost import XGBRegressor
import time

seeds = [42, 7, 2026, 11, 99, 5, 17, 23, 31, 53]
models = []
for s in seeds:
    t0 = time.time()
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5,
                     tree_method='hist', device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
    m.fit(Xt, yt)
    models.append(m)
    print(f"  seed={s} trained in {time.time()-t0:.1f}s")

print("Saving 10-model ensemble to model.pkl...")
os.makedirs("/workspace/pretrained", exist_ok=True)
with open("/workspace/pretrained/model.pkl", "wb") as f:
    pickle.dump((models, feat, "target"), f)
print(f"Saved. Total size: {os.path.getsize('/workspace/pretrained/model.pkl')/1e6:.1f} MB")
