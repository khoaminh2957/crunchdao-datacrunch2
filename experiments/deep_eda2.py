"""Deep EDA continued — feature distribution, correlations, target class."""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet("C:/Users/Admin/Downloads/X.reduced.parquet")
y = pd.read_parquet("C:/Users/Admin/Downloads/y.reduced.parquet")
feat_cols = [c for c in X.columns if c not in ('id','moon')]

print("="*70)
print("PHASE 7: FEATURE DISTRIBUTION SHAPE")
print("="*70)
sample = X[feat_cols[:3] + feat_cols[574:577]].sample(50000, random_state=42)
for c in sample.columns:
    s = sample[c]
    print(f"  {c}: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}, "
          f"q01={s.quantile(0.01):.4f}, q99={s.quantile(0.99):.4f}")

print("\n"+"="*70)
print("PHASE 8: TARGET as 3-CLASS")
print("="*70)
t = y['target']
neg = (t < -0.5).sum(); pos = (t > 0.5).sum(); zero = (t == 0).sum()
mid = len(t) - neg - pos - zero
print(f"target == -1 (or near): {neg} ({neg/len(t)*100:.2f}%)")
print(f"target ==  0          : {zero} ({zero/len(t)*100:.2f}%)")
print(f"target ==  1 (or near): {pos} ({pos/len(t)*100:.2f}%)")
print(f"target middle values  : {mid} ({mid/len(t)*100:.2f}%)")
print(f"\nUnique target values (sample): {sorted(t.unique())[:5]} ... {sorted(t.unique())[-5:]}")
print(f"Total unique target values: {t.nunique()}")

print("\n"+"="*70)
print("PHASE 9: INTER-FEATURE CORRELATION (sample 50 features)")
print("="*70)
samp_feats = feat_cols[::23][:50]  # spread across feature space
sample_X = X[samp_feats].sample(50000, random_state=42)
corr_mat = sample_X.corr().abs()
np.fill_diagonal(corr_mat.values, 0)
print(f"Max abs inter-feature corr: {corr_mat.values.max():.4f}")
print(f"Mean abs inter-feature corr: {corr_mat.values.mean():.4f}")
print(f"Features with max corr > 0.5: {(corr_mat.max() > 0.5).sum()}")
print(f"Features with max corr > 0.9: {(corr_mat.max() > 0.9).sum()}")

print("\n"+"="*70)
print("PHASE 10: PER-MOON CORRELATION of target with features")
print("="*70)
# Sample 100 moons, compute per-moon correlation for first 20 features
moon_sample = X['moon'].sample(100, random_state=42).unique()[:50]
yf = X[['id','moon']+feat_cols[:20]].merge(y, on=['id','moon'])
per_moon_corrs = []
for c in feat_cols[:20]:
    pmc = yf.groupby('moon').apply(lambda d: d[c].corr(d['target']))
    per_moon_corrs.append({
        'feat': c,
        'mean': pmc.mean(),
        'std': pmc.std(),
        'min': pmc.min(),
        'max': pmc.max(),
        'pct_pos': (pmc > 0).mean(),
    })
print(f"Per-moon corr stats (first 20 features):")
print(pd.DataFrame(per_moon_corrs).round(4).to_string(index=False))

print("\n"+"="*70)
print("PHASE 11: BEST FEATURES by overall correlation")
print("="*70)
# Compute corr for ALL features using a SAMPLE of 200k rows
print("Computing on 200k row sample...")
samp = X[['id','moon']+feat_cols].sample(200000, random_state=42)
merged = samp.merge(y, on=['id','moon'])
corrs = {}
for c in feat_cols:
    corrs[c] = merged[c].corr(merged['target'])
corr_ser = pd.Series(corrs).sort_values(key=lambda s: s.abs(), ascending=False)
print(f"\nTop 15 features by |corr|:")
print(corr_ser.head(15).round(5).to_string())
print(f"\nBottom 10 features by |corr|:")
print(corr_ser.tail(10).round(5).to_string())
print(f"\nCorr distribution: min={corr_ser.min():.4f}, max={corr_ser.max():.4f}, "
      f"|mean|={corr_ser.abs().mean():.4f}, |median|={corr_ser.abs().median():.4f}")
