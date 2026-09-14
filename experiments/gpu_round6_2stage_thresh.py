"""Round 6: 2-stage with different non-zero thresholds.

Default uses |target| > 0.5 → 9.4% non-zero.
Test thresholds: 0.05, 0.1, 0.2, 0.5 — see which gives best lift on k=9.
"""
import pandas as pd, numpy as np, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor, XGBClassifier

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

# Test all thresholds on 5 train_ends × test moon (T+9)
TRAIN_ENDS = [600, 650, 700, 750, 770]
THRESHOLDS = [0.05, 0.1, 0.2, 0.5]

print(f"{'TrainEnd':<8} | {'Test moon':<10} | {'Baseline':<10} | thresh: " +
      " ".join([f"{t:>6}" for t in THRESHOLDS]))
print('-' * 100)

all_results = {t: [] for t in THRESHOLDS}
baseline_results = []

for T in TRAIN_ENDS:
    tm = T + 9
    if tm > merged['moon'].max():
        continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test_sub = merged[merged['moon'] == tm]
    Xte = test_sub[feat].values.astype(np.float32)
    y_te = test_sub['target'].values

    # Baseline
    m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                       device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
    m_b.fit(Xt, yt)
    p_b = m_b.predict(Xte)
    r_b = pearsonr(p_b, y_te)[0]
    baseline_results.append(r_b)

    line = f"T={T:<5} | moon {tm:<5} | {r_b:+.5f} |"
    for thr in THRESHOLDS:
        y_nz = (np.abs(yt) > thr).astype(np.int32)
        if y_nz.sum() < 100:
            line += f" {0:>6}"
            all_results[thr].append(0)
            continue
        m_c = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                            device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m_c.fit(Xt, y_nz)
        p_c = m_c.predict_proba(Xte)[:, 1]
        # Stage 2: regressor on non-zero (using same threshold)
        mask_nz = y_nz == 1
        m_r = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m_r.fit(Xt[mask_nz], yt[mask_nz])
        p_r = m_r.predict(Xte)
        p_comb = p_c * p_r
        r = pearsonr(p_comb, y_te)[0]
        all_results[thr].append(r)
        line += f" {r:+.4f}"
    print(line)

print(f"\n=== Mean across {len(TRAIN_ENDS)} train_ends ===")
print(f"Baseline (single XGB):     mean={np.mean(baseline_results):+.5f}")
for thr in THRESHOLDS:
    if all_results[thr]:
        print(f"2-stage thr={thr}:           mean={np.mean(all_results[thr]):+.5f}  lift={np.mean(all_results[thr])-np.mean(baseline_results):+.5f}")
