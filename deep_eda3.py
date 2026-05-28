"""Phase 12-14: classification framing + feature group analysis."""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet("C:/Users/Admin/Downloads/X.reduced.parquet")
y = pd.read_parquet("C:/Users/Admin/Downloads/y.reduced.parquet")
feat_cols = [c for c in X.columns if c not in ('id','moon')]

print("="*70)
print("PHASE 12: ALL features full-data correlation (find groups)")
print("="*70)
print("Computing all 1150 feature-target corrs on full data...")
merged = X.merge(y, on=['id','moon'], how='inner')
target = merged['target'].values
corrs = pd.Series({c: merged[c].corr(merged['target']) for c in feat_cols})
print(f"Distribution: min={corrs.min():.4f}, max={corrs.max():.4f}")
print(f"\nTop 30 by |corr|:")
top30 = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(30)
for k,v in top30.items():
    print(f"  {k}: {v:+.5f}")

print("\n"+"="*70)
print("PHASE 13: Are features SORTED by signal? (Feature_N vs |corr|)")
print("="*70)
# Plot correlation by feature index
idx = [int(c.split('_')[1]) for c in feat_cols]
df = pd.DataFrame({'idx': idx, 'corr': corrs.values}).sort_values('idx')
# Bin by 50 features
df['bin'] = (df['idx']-1) // 50
binned = df.groupby('bin').agg(min_idx=('idx','min'), max_idx=('idx','max'),
                                 mean_abs_corr=('corr', lambda s: s.abs().mean()),
                                 max_abs_corr=('corr', lambda s: s.abs().max())).round(5)
print("Features binned by 50:")
print(binned.head(15).to_string())
print("...")
print(binned.tail(5).to_string())

print("\n"+"="*70)
print("PHASE 14: Predict TARGET SIGN as classification")
print("="*70)
# Convert target to 3-class: -1 if <-0.5, +1 if >0.5, 0 otherwise
y_class = np.where(target < -0.5, -1, np.where(target > 0.5, 1, 0))
print(f"Class distribution: -1={np.mean(y_class==-1)*100:.2f}%, 0={np.mean(y_class==0)*100:.2f}%, +1={np.mean(y_class==1)*100:.2f}%")

# Top features classification corr (Spearman with class)
print("\nTop 10 features by Pearson corr with sign class:")
sign_corr = {}
for c in feat_cols[:200]:  # sample to speed up
    sign_corr[c] = pd.Series(merged[c].values).corr(pd.Series(y_class))
sc = pd.Series(sign_corr)
sc_sorted = sc.reindex(sc.abs().sort_values(ascending=False).index).head(15)
print(sc_sorted.round(5).to_string())

print("\n"+"="*70)
print("PHASE 15: SPEARMAN per-moon corr (matches scorer)")
print("="*70)
# For top 5 features, compute Spearman corr per moon
top5 = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(5).index.tolist()
print(f"Top 5 features for per-moon Spearman test: {top5}")
for c in top5:
    sub = merged[[c, 'target', 'moon']]
    from scipy.stats import spearmanr
    spearmen = sub.groupby('moon').apply(lambda d: spearmanr(d[c], d['target'])[0] if len(d) > 5 else np.nan, include_groups=False)
    print(f"  {c}: per-moon Spearman mean={spearmen.mean():.4f}, std={spearmen.std():.4f}, "
          f"q25={spearmen.quantile(0.25):.4f}, q75={spearmen.quantile(0.75):.4f}")
