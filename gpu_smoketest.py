"""Single-fold GPU smoke test: 1 XGB train on real data, time it, score it.

Used to validate GPU speed + memory before kicking off 32-config sweep.
"""
import pandas as pd
import numpy as np
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr

print("Loading data...")
t0 = time.time()
X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
print(f"  X={X.shape}, y={y.shape} loaded in {time.time()-t0:.1f}s")

feat = [c for c in X.columns if c not in ('id','moon')]
T = 670  # train ends at moon 670, test on 671-679 (9-moon horizon like cloud)

tr_mask = X['moon'] <= T
te_mask = (X['moon'] > T) & (X['moon'] <= T + 9)
X_tr = X[tr_mask].reset_index(drop=True)
X_te = X[te_mask].reset_index(drop=True)
y_tr = y[y['moon'] <= T].reset_index(drop=True)
y_te = y[(y['moon'] > T) & (y['moon'] <= T + 9)].reset_index(drop=True)
print(f"  train: {len(X_tr)} rows, test: {len(X_te)} rows ({y_te['moon'].nunique()} moons)")

# Per-moon rank target
def rank_per_moon(df):
    grp = df.groupby('moon')['target']
    return (grp.rank(method='average') / grp.transform('count')).astype(np.float32)

t1 = time.time()
merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
mask = merged['target'].notna()
merged = merged.loc[mask].reset_index(drop=True)
y_ranked = rank_per_moon(merged[['moon','target']]).values
Xt = merged[feat].values.astype(np.float32)
print(f"  merge + rank: {time.time()-t1:.1f}s; Xt={Xt.shape}, yt={y_ranked.shape}")

print("\nTraining XGB (500/d6/lr0.03 — v14 config) on GPU 0...")
from xgboost import XGBRegressor
t2 = time.time()
model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.5,
    tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
)
model.fit(Xt, y_ranked)
print(f"  XGB GPU train: {time.time()-t2:.1f}s")

t3 = time.time()
preds = model.predict(X_te[feat].values.astype(np.float32))
print(f"  Infer: {time.time()-t3:.1f}s")

# Score
pred_df = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': preds})
merged_eval = pred_df.merge(y_te, on=['id','moon'], how='inner')
scores = []
for m, sub in merged_eval.groupby('moon'):
    r = spearmanr(sub['prediction'], sub['target'])[0]
    if np.isfinite(r):
        scores.append(r)
print(f"\n  Per-moon Spearman: mean={np.mean(scores):+.5f}, std={np.std(scores):.5f}, n={len(scores)} moons")
print(f"  Per-moon scores: {[round(s,4) for s in scores]}")
print(f"\nTotal smoke test: {time.time()-t0:.1f}s")
