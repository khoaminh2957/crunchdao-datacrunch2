"""Tiny v52-equivalent validation: n_est=50, depth=4, moons 700-708 holdout."""
import pandas as pd, numpy as np
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr

X = pd.read_parquet(r"C:\Users\Admin\Downloads\X.reduced.parquet")
y = pd.read_parquet(r"C:\Users\Admin\Downloads\y.reduced.parquet")
print(f"X shape {X.shape}, y shape {y.shape}")
print(f"X cols sample: {list(X.columns[:6])}")
print(f"y cols: {list(y.columns)}")
print(f"moons in X: min={X['moon'].min()} max={X['moon'].max()} unique={X['moon'].nunique()}")

# join
join_cols = [c for c in ("id", "moon") if c in X.columns and c in y.columns]
merged = X.merge(y, on=join_cols, how="inner")
target_col = "target" if "target" in merged.columns else [c for c in y.columns if c not in ("id","moon")][0]
print(f"target_col={target_col}, merged={merged.shape}")

feature_cols = [c for c in X.columns if c not in ("id","Id","moon","Moon")]
print(f"n features = {len(feature_cols)}")

HOLDOUT = list(range(700, 709))
train_mask = (~merged["moon"].isin(HOLDOUT)) & merged[target_col].notna()
test_mask = merged["moon"].isin(HOLDOUT)
print(f"train rows={train_mask.sum()}, test rows={test_mask.sum()}")
print(f"test moons present: {sorted(merged.loc[test_mask,'moon'].unique().tolist())}")

train_df = merged.loc[train_mask].reset_index(drop=True)
# per-moon rank
grp = train_df.groupby("moon")[target_col]
y_ranked = (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32).values
print(f"y_ranked range [{y_ranked.min():.4f}, {y_ranked.max():.4f}] mean={y_ranked.mean():.4f}")

Xt = train_df[feature_cols]
model = XGBRegressor(
    n_estimators=50, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.5,
    tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
)
print("fitting tiny model ...")
model.fit(Xt, y_ranked)
print("done fit")

test_df = merged.loc[test_mask].reset_index(drop=True)
preds = model.predict(test_df[feature_cols])
print(f"\npred range [{preds.min():.6f}, {preds.max():.6f}] mean={preds.mean():.6f} std={preds.std():.6f}")
print(f"NaN preds: {np.isnan(preds).sum()}")
print(f"constant pred check: unique values approx = {len(np.unique(np.round(preds,5)))}")

print("\nper-moon stats:")
print(f"{'moon':>6} {'n':>6} {'pred_std':>10} {'pearson':>10} {'spearman':>10}")
for m in sorted(test_df["moon"].unique()):
    mask = test_df["moon"] == m
    p = preds[mask.values]
    t = test_df.loc[mask, target_col].values
    valid = ~np.isnan(t)
    if valid.sum() < 5:
        print(f"{m:>6} {mask.sum():>6} {p.std():>10.6f} (target all NaN or too few)")
        continue
    pr = pearsonr(p[valid], t[valid])[0] if p[valid].std() > 0 else 0.0
    sr = spearmanr(p[valid], t[valid])[0] if p[valid].std() > 0 else 0.0
    print(f"{m:>6} {valid.sum():>6} {p.std():>10.6f} {pr:>10.4f} {sr:>10.4f}")
