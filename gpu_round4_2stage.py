"""Round 4: 2-stage model.

Stage 1: XGBClassifier predicts P(non-zero) on all rows.
Stage 2: XGBRegressor trained ONLY on non-zero rows, predicts magnitude/sign.
Combine: pred = P(non-zero) * regressor_pred.

Compare against single XGB on full data.
"""
import pandas as pd, numpy as np, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor, XGBClassifier

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

print("Training on moons 1-770, testing on moon 779 (k=9, mimics cloud)...")
tr = merged[merged['moon'] <= 770]
te = merged[merged['moon'] == 779]
yt = tr['target'].values.astype(np.float32)
Xt = tr[feat].values.astype(np.float32)
Xte = te[feat].values.astype(np.float32)
y_te = te['target'].values

# === Baseline: single regressor ===
print("\n=== A. Baseline single XGBRegressor ===")
t0 = time.time()
m_baseline = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
m_baseline.fit(Xt, yt)
p_baseline = m_baseline.predict(Xte)
r_baseline = pearsonr(p_baseline, y_te)[0]
print(f"  Pearson on moon 779: {r_baseline:+.5f} ({time.time()-t0:.0f}s)")

# === 2-stage approach ===
print("\n=== B. 2-stage: classifier(non-zero) × regressor(non-zero) ===")
# Stage 1: binary classifier on all rows
y_nonzero = (np.abs(yt) > 0.5).astype(np.int32)
print(f"  Non-zero ratio in train: {y_nonzero.mean()*100:.1f}%")
t0 = time.time()
m_cls = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                      device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
m_cls.fit(Xt, y_nonzero)
p_nonzero = m_cls.predict_proba(Xte)[:, 1]
print(f"  Stage 1 trained in {time.time()-t0:.0f}s")
# Stage 2: regressor on non-zero only
mask_nz = np.abs(yt) > 0.5
Xt_nz = Xt[mask_nz]; yt_nz = yt[mask_nz]
print(f"  Stage 2 training set: {len(Xt_nz)} rows")
t0 = time.time()
m_reg = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
m_reg.fit(Xt_nz, yt_nz)
p_reg = m_reg.predict(Xte)
print(f"  Stage 2 trained in {time.time()-t0:.0f}s")

# Combine
p_combined = p_nonzero * p_reg
r_combined = pearsonr(p_combined, y_te)[0]
print(f"  Combined pred Pearson: {r_combined:+.5f}")
print(f"  Stage 1 alone (P(non-zero) vs target): {pearsonr(p_nonzero, y_te)[0]:+.5f}")
print(f"  Stage 2 alone (sign/magnitude vs target): {pearsonr(p_reg, y_te)[0]:+.5f}")

# === Average baseline + 2-stage ===
print("\n=== C. Average baseline + 2-stage ===")
p_avg = (p_baseline + p_combined) / 2
print(f"  Pearson: {pearsonr(p_avg, y_te)[0]:+.5f}")

# === Test on multiple k horizons ===
print("\n=== D. Test 2-stage on moons 772-781 ===")
print(f"{'Moon':<6} | {'k':<3} | {'baseline':<10} | {'2-stage':<10} | {'2-stage > baseline?':<20}")
print('-' * 60)
for tm in range(772, 782):
    sub = merged[merged['moon'] == tm]
    if len(sub) < 100: continue
    Xte = sub[feat].values.astype(np.float32)
    y_te = sub['target'].values
    p_b = m_baseline.predict(Xte)
    p_c = m_cls.predict_proba(Xte)[:, 1]
    p_r = m_reg.predict(Xte)
    p_comb = p_c * p_r
    r_b = pearsonr(p_b, y_te)[0]
    r_c = pearsonr(p_comb, y_te)[0]
    winner = "✓" if r_c > r_b else " "
    print(f"{tm:<6} | {tm-770:<3} | {r_b:+.5f}   | {r_c:+.5f}   | {winner}")
