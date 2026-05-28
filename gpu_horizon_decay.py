"""TEST 'signal decay' hypothesis: Pearson on T+k for k = 1, 2, 3, 5, 7, 9.

If signal decays, Pearson should drop monotonically with k.
This explains why cloud score (k=9, last moon) is much lower than k=1 results.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

# For TRAIN ends T = 600, 650, 700, train and test on T+k for various k
TRAIN_ENDS = [600, 650, 700, 760]
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20]

from xgboost import XGBRegressor

print("Horizon-decay test:")
print(f"{'TrainEnd':<8} | ", end="")
for h in HORIZONS:
    print(f"k={h:>2}    ", end="")
print()
print("-" * 130)

results = []
for T in TRAIN_ENDS:
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    t0 = time.time()
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                     colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt)
    train_t = time.time() - t0
    print(f"T={T:<5} | ", end="")
    for h in HORIZONS:
        test_m = T + h
        if test_m > merged['moon'].max():
            print(f"  N/A   ", end="")
            continue
        sub = merged[merged['moon'] == test_m]
        pred = m.predict(sub[feat].values.astype(np.float32))
        if sub['target'].std() < 1e-10 or pred.std() < 1e-10:
            r = 0
        else:
            r = pearsonr(pred, sub['target'].values)[0]
        print(f"{r:+.4f}  ", end="")
        results.append({'T': T, 'horizon': h, 'pearson': r})
    print(f"  (train {train_t:.0f}s)")

# Summary
import json
with open('/workspace/results/horizon_decay.json', 'w') as f:
    json.dump(results, f, indent=2)

# Mean per horizon
print("\n=== Mean Pearson per horizon (across train ends) ===")
df = pd.DataFrame(results)
agg = df.groupby('horizon').agg(mean=('pearson','mean'), std=('pearson','std'))
print(agg.round(4).to_string())
