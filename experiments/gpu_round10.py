"""R10: more loss/objective experiments.

Test:
A. Quantile regression (median)
B. Sample weight TUNED (2, 5)
C. Huber DELTA parameter sweep (0.5, 1, 2)
D. Different bins of target (binary >0)
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor, XGBClassifier

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')
TRAIN_ENDS = [600, 650, 700, 750, 770]
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'

def k9(name, model_fn):
    scores = []
    for T in TRAIN_ENDS:
        tm = T + 9
        if tm > merged['moon'].max(): continue
        tr = merged[merged['moon'] <= T]
        yt = tr['target'].values.astype(np.float32)
        Xt = tr[feat].values.astype(np.float32)
        test = merged[merged['moon'] == tm]
        Xte = test[feat].values.astype(np.float32)
        y_te = test['target'].values
        t0 = time.time()
        pred = model_fn(Xt, yt, Xte)
        r = pearsonr(pred, y_te)[0]
        scores.append(r)
        print(f"  [{name}] T={T} pearson={r:+.5f} ({time.time()-t0:.0f}s)", flush=True)
    return scores


# A. Quantile regression
def quantile_med(Xt, yt, Xte):
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     objective='reg:quantileerror', quantile_alpha=0.5,
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt); return m.predict(Xte)
print("=== A. Quantile median ==="); A = k9('quantile', quantile_med)

# B1. sample weight x2
def sw2(Xt, yt, Xte):
    w = np.where(np.abs(yt) > 0.5, 2.0, 1.0).astype(np.float32)
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt, sample_weight=w); return m.predict(Xte)
print("\n=== B1. Sample weight ×2 ==="); B1 = k9('sw=2', sw2)

# B2. sample weight x5
def sw5(Xt, yt, Xte):
    w = np.where(np.abs(yt) > 0.5, 5.0, 1.0).astype(np.float32)
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt, sample_weight=w); return m.predict(Xte)
print("\n=== B2. Sample weight ×5 ==="); B2 = k9('sw=5', sw5)

# C. Huber alt delta? XGB doesn't expose huber delta directly
# D. Binary >0 target predictor
def binary_pos(Xt, yt, Xte):
    y_bin = (yt > 0).astype(np.int32)
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, y_bin); return m.predict_proba(Xte)[:, 1]
print("\n=== D. Binary >0 classifier ==="); D = k9('bin>0', binary_pos)

print("\n=== R10 SUMMARY ===")
for nm, sc in [('quantile_med', A), ('sw=2', B1), ('sw=5', B2), ('bin>0', D)]:
    if sc: print(f"  {nm:<20} mean={np.mean(sc):+.5f} std={np.std(sc):.5f}")
