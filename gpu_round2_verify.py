"""Round 2 EDA: explicitly verify multiple claims from Round 1 with new measurements.

Tests:
A. Feature_1 vs Feature_43 — are they REALLY identical?
B. Feature_756/757/758 (industry candidate per A8 agent) — do they correlate with target?
C. Top 30 features by ACTUAL correlation (verify A8 claims)
D. Predictability of single moon vs ensemble of moons
"""
import pandas as pd, numpy as np, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr, spearmanr

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
target = merged['target'].values

# === A. Feature_1 vs Feature_43 — identity check ===
print("=== A. Feature_1 vs Feature_43 identity ===")
diff = merged['Feature_1'] - merged['Feature_43']
print(f"  mean diff: {diff.mean():.6f}")
print(f"  max abs diff: {diff.abs().max():.6f}")
print(f"  Pearson(F1, F43): {pearsonr(merged['Feature_1'], merged['Feature_43'])[0]:.6f}")
print(f"  Exact equal rows: {(diff == 0).mean()*100:.2f}%")

# === B. Industry candidate features (per A8) ===
print("\n=== B. A8's industry candidate Feature_756/757/758 vs target ===")
for f in ['Feature_756', 'Feature_757', 'Feature_758', 'Feature_1037', 'Feature_1038', 'Feature_1076']:
    r = pearsonr(merged[f], target)[0]
    print(f"  {f}: corr with target = {r:+.5f}")

# === C. Top 30 features by ACTUAL Pearson ===
print("\n=== C. Top 30 features by |Pearson with target| ===")
print("Computing all 1150 correlations...")
t0 = time.time()
corrs = {f: pearsonr(merged[f], target)[0] for f in feat}
print(f"  done in {time.time()-t0:.1f}s")
df = pd.Series(corrs).reindex(pd.Series(corrs).abs().sort_values(ascending=False).index)
print(df.head(30).round(5).to_string())
print(f"\nVerify claim 'top 14 are 2 duplicate groups (1-7 and 43-49)':")
top14 = df.head(14).index.tolist()
g_1to7 = sum(1 for f in top14 if f in [f"Feature_{i}" for i in range(1, 8)])
g_43to49 = sum(1 for f in top14 if f in [f"Feature_{i}" for i in range(43, 50)])
print(f"  Top 14 contains {g_1to7} from Feature_1-7 and {g_43to49} from Feature_43-49")

# === D. Per-moon Pearson distribution for TOP feature on test data ===
print("\n=== D. Single-feature Feature_1 per-moon Pearson — distribution ===")
per_moon_corr = []
for m in sorted(merged['moon'].unique()):
    sub = merged[merged['moon'] == m]
    if len(sub) < 100: continue
    r = pearsonr(sub['Feature_1'], sub['target'])[0]
    per_moon_corr.append(r)
per_moon_corr = np.array(per_moon_corr)
print(f"  n_moons={len(per_moon_corr)}")
print(f"  mean={per_moon_corr.mean():.5f} (overall corr was -0.0232)")
print(f"  std={per_moon_corr.std():.5f}")
print(f"  min={per_moon_corr.min():.5f}, max={per_moon_corr.max():.5f}")
print(f"  pct_positive={(per_moon_corr > 0).mean()*100:.1f}%  (expect <50% if true corr is negative)")
print(f"  Pct |r| < 0.025 (noise floor): {(np.abs(per_moon_corr) < 0.025).mean()*100:.1f}%")

# === E. SIGN baseline ===
print("\n=== E. Sign-only Pearson per moon — actual not hypothetical ===")
sign_corrs = []
for m in sorted(merged['moon'].unique()):
    sub = merged[merged['moon'] == m]
    if len(sub) < 100: continue
    sign = np.sign(sub['target'].values)
    if sign.std() < 1e-10: continue
    r = pearsonr(sign, sub['target'].values)[0]
    sign_corrs.append(r)
sign_corrs = np.array(sign_corrs)
print(f"  Sign(target) Pearson per moon: mean={sign_corrs.mean():.4f}, std={sign_corrs.std():.4f}")
print(f"  This is the THEORETICAL ceiling if classifier predicts sign perfectly.")
