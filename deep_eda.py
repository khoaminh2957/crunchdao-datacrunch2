"""Deep EDA on the reduced dataset."""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

X_PATH = "C:/Users/Admin/Downloads/X.reduced.parquet"
Y_PATH = "C:/Users/Admin/Downloads/y.reduced.parquet"

print("="*70)
print("PHASE 1: SHAPE + SCHEMA")
print("="*70)
X = pd.read_parquet(X_PATH)
y = pd.read_parquet(Y_PATH)
print(f"X: {X.shape}, mem={X.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"y: {y.shape}, mem={y.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"\nX columns sample (first 10 + last 5):")
print(list(X.columns[:10]) + ['...'] + list(X.columns[-5:]))
print(f"X dtypes (counts): {X.dtypes.value_counts().to_dict()}")
print(f"\ny columns: {list(y.columns)}")
print(f"y dtypes: {y.dtypes.to_dict()}")

print("\n"+"="*70)
print("PHASE 2: TIME (moon) DISTRIBUTION")
print("="*70)
print(f"X moon range: [{X['moon'].min()}, {X['moon'].max()}], n_unique={X['moon'].nunique()}")
print(f"y moon range: [{y['moon'].min()}, {y['moon'].max()}], n_unique={y['moon'].nunique()}")
counts = X.groupby('moon').size()
print(f"Rows per moon: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}, mean={counts.mean():.0f}")
print(f"First 5 moons sizes: {counts.head().to_dict()}")
print(f"Last 5 moons sizes: {counts.tail().to_dict()}")

print("\n"+"="*70)
print("PHASE 3: TARGET (y)")
print("="*70)
y_target = y.iloc[:, -1] if y.columns[-1] not in ('id','moon') else y['target']
print(f"Target column: '{y.columns[-1]}'")
print(f"Stats: min={y_target.min():.6f}, max={y_target.max():.6f}, mean={y_target.mean():.6f}, std={y_target.std():.6f}")
print(f"NaN: {y_target.isna().sum()} ({y_target.isna().mean()*100:.2f}%)")
print(f"== 0: {(y_target==0).sum()} ({(y_target==0).mean()*100:.2f}%)")
print(f"Quantiles: 1%={y_target.quantile(0.01):.4f}, 25%={y_target.quantile(0.25):.4f}, 50%={y_target.quantile(0.5):.4f}, 75%={y_target.quantile(0.75):.4f}, 99%={y_target.quantile(0.99):.4f}")
print(f"\nPer-moon target stats (first 5 moons):")
y_per_moon = y.groupby('moon')[y.columns[-1]].agg(['count','mean','std','min','max'])
print(y_per_moon.head().round(4))

print("\n"+"="*70)
print("PHASE 4: FEATURES — NaN / variance / range")
print("="*70)
feat_cols = [c for c in X.columns if c not in ('id','moon','Id','Moon')]
print(f"Total features: {len(feat_cols)}")
nan_pct = X[feat_cols].isna().mean()
print(f"NaN per feature: min={nan_pct.min()*100:.2f}%, mean={nan_pct.mean()*100:.2f}%, max={nan_pct.max()*100:.2f}%")
print(f"Features with NaN > 50%: {(nan_pct > 0.5).sum()}")
print(f"Features all-NaN: {(nan_pct == 1.0).sum()}")
var_per_feat = X[feat_cols].var(numeric_only=True)
print(f"Variance: min={var_per_feat.min():.6f}, mean={var_per_feat.mean():.4f}, max={var_per_feat.max():.4f}")
print(f"Constant features (var=0): {(var_per_feat == 0).sum()}")
print(f"Features with var < 1e-6: {(var_per_feat < 1e-6).sum()}")

print("\n"+"="*70)
print("PHASE 5: ID — uniqueness, persistence across moons")
print("="*70)
print(f"Unique ids in X: {X['id'].nunique()}")
ids_per_moon = X.groupby('moon')['id'].nunique()
print(f"Unique ids per moon: min={ids_per_moon.min()}, median={ids_per_moon.median():.0f}, max={ids_per_moon.max()}")
# Persistence: how many moons does an id appear in?
id_moon_count = X.groupby('id')['moon'].nunique()
print(f"Moons per id: min={id_moon_count.min()}, median={id_moon_count.median():.0f}, max={id_moon_count.max()}, mean={id_moon_count.mean():.1f}")
print(f"IDs in 1 moon only: {(id_moon_count == 1).sum()} ({(id_moon_count == 1).mean()*100:.1f}%)")
print(f"IDs in >=100 moons: {(id_moon_count >= 100).sum()} ({(id_moon_count >= 100).mean()*100:.1f}%)")

print("\n"+"="*70)
print("PHASE 6: CORRELATION sample (first 5 features vs target after merge)")
print("="*70)
merged = X[['id','moon'] + feat_cols[:5]].merge(y, on=['id','moon'], how='inner')
print(f"Merged shape: {merged.shape}")
for c in feat_cols[:5]:
    sub = merged[[c, y.columns[-1]]].dropna()
    if len(sub) > 100:
        corr = sub[c].corr(sub[y.columns[-1]])
        print(f"  {c}: corr={corr:.4f} (n={len(sub)})")
