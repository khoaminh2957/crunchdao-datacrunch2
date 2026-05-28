"""Local validation: train on first 700 moons, evaluate on moons 701-781 (holdout).
This mimics the cloud test setup where train ends at moon 781 and test is 782+.
"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr

print("Loading data...")
X = pd.read_parquet("C:/Users/Admin/Downloads/X.reduced.parquet")
y = pd.read_parquet("C:/Users/Admin/Downloads/y.reduced.parquet")
print(f"X={X.shape}, y={y.shape}")

# Split: train on moons 1-700, holdout 701-781
TRAIN_MAX_MOON = 700
train_mask = X['moon'] <= TRAIN_MAX_MOON
test_mask = X['moon'] > TRAIN_MAX_MOON

X_tr = X[train_mask].reset_index(drop=True)
X_te = X[test_mask].reset_index(drop=True)
y_tr = y[y['moon'] <= TRAIN_MAX_MOON].reset_index(drop=True)
y_te = y[y['moon'] > TRAIN_MAX_MOON].reset_index(drop=True)
print(f"Train: {len(X_tr)} rows, moons {X_tr['moon'].min()}-{X_tr['moon'].max()}")
print(f"Test : {len(X_te)} rows, moons {X_te['moon'].min()}-{X_te['moon'].max()}")


def rank_per_moon(y_df, moon_col='moon', target_col='target'):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def per_moon_spearman(pred_df, truth_df):
    """Compute mean per-moon Spearman correlation (matches CrunchDAO scorer)."""
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    scores = []
    for m, sub in merged.groupby('moon'):
        if len(sub) < 5: continue
        r = spearmanr(sub['prediction'], sub['target'])[0]
        if np.isfinite(r):
            scores.append(r)
    return np.mean(scores), len(scores)


# Common train logic
def train_xgb(X_train, feature_cols, y_train, **xgb_kwargs):
    from xgboost import XGBRegressor
    merged = X_train[['id','moon']+feature_cols].merge(y_train, on=['id','moon'], how='inner')
    mask = merged['target'].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    y_ranked = rank_per_moon(merged[['moon','target']]).values
    Xt = merged[feature_cols]
    params = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.5,
                  tree_method='hist', device='cuda', n_jobs=-1, random_state=42, verbosity=0)
    params.update(xgb_kwargs)
    model = XGBRegressor(**params)
    print(f"  Training XGB on {len(Xt)} rows × {len(feature_cols)} feats")
    model.fit(Xt, y_ranked)
    return model


def infer_xgb(model, X_test, feature_cols):
    preds = model.predict(X_test[feature_cols])
    return pd.DataFrame({'id': X_test['id'].values, 'moon': X_test['moon'].values, 'prediction': preds})


print("\n"+"="*60)
print("V14 BASELINE: all 1150 features")
print("="*60)
feat_v14 = [c for c in X_tr.columns if c not in ('id','moon')]
m = train_xgb(X_tr, feat_v14, y_tr)
pred = infer_xgb(m, X_te, feat_v14)
score, n = per_moon_spearman(pred, y_te)
print(f"  → Per-moon Spearman: {score:.4f} (n={n} moons)")

print("\n"+"="*60)
print("V41: drop duplicates (Feature_43-49)")
print("="*60)
feat_v41 = [c for c in feat_v14 if c not in [f"Feature_{i}" for i in range(43,50)]]
m = train_xgb(X_tr, feat_v41, y_tr)
pred = infer_xgb(m, X_te, feat_v41)
score, n = per_moon_spearman(pred, y_te)
print(f"  → Per-moon Spearman: {score:.4f}")

print("\n"+"="*60)
print("V40: top 100 features by |corr|")
print("="*60)
TOP_100 = [
    "Feature_1", "Feature_43", "Feature_44", "Feature_2", "Feature_3",
    "Feature_45", "Feature_46", "Feature_4", "Feature_47", "Feature_5",
    "Feature_48", "Feature_6", "Feature_7", "Feature_49", "Feature_83",
    "Feature_87", "Feature_99", "Feature_1090", "Feature_1087", "Feature_68",
    "Feature_1088", "Feature_1089", "Feature_64", "Feature_1124", "Feature_21",
    "Feature_1096", "Feature_483", "Feature_1127", "Feature_249", "Feature_489",
] + [f"Feature_{i}" for i in [80,81,84,85,86,88,89,90,91,92,60,61,62,63,65,66,67,69,70,8,
                              9,10,11,12,13,14,15,50,51,52,53,54,55,56,57,58,59,73,74,75,
                              76,77,78,79,100,101,102,103,104,105,200,201,250,251,
                              482,484,485,486,487,488,1085,1086,1091,1092,1093,1094,1095,1097,1098,1099]]
feat_v40 = [c for c in TOP_100 if c in feat_v14][:100]
m = train_xgb(X_tr, feat_v40, y_tr)
pred = infer_xgb(m, X_te, feat_v40)
score, n = per_moon_spearman(pred, y_te)
print(f"  → Per-moon Spearman: {score:.4f}")

print("\n"+"="*60)
print("V40-tight: TOP 14 features only (the 2 duplicate groups)")
print("="*60)
feat_tight = [f"Feature_{i}" for i in [1,2,3,4,5,6,7,43,44,45,46,47,48,49]]
feat_tight = [c for c in feat_tight if c in feat_v14]
m = train_xgb(X_tr, feat_tight, y_tr)
pred = infer_xgb(m, X_te, feat_tight)
score, n = per_moon_spearman(pred, y_te)
print(f"  → Per-moon Spearman: {score:.4f}")

print("\n"+"="*60)
print("V41b: drop duplicates + reduce to top 50 by corr")
print("="*60)
feat_top50_no_dup = [c for c in feat_v40[:50] if c not in [f"Feature_{i}" for i in range(43,50)]]
m = train_xgb(X_tr, feat_top50_no_dup, y_tr)
pred = infer_xgb(m, X_te, feat_top50_no_dup)
score, n = per_moon_spearman(pred, y_te)
print(f"  → Per-moon Spearman: {score:.4f}")
