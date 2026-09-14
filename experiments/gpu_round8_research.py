"""Round 8: parallel research suite on cuda:1.

Each experiment tests ONE hypothesis. Results saved incrementally.
After each: claim made, will need verification.
"""
import pandas as pd, numpy as np, time, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor, XGBClassifier

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

# Common test: T=650, test moon = 659 (k=9)
TRAIN_ENDS = [600, 650, 700, 750, 770]
HORIZON_K = 9
DEV = 'cuda:1'
RES = {}

def k9_eval(name, model_fn):
    """Train model_fn on each train_end, eval on moon T+9. Return list of Pearsons."""
    scores = []
    for T in TRAIN_ENDS:
        tm = T + HORIZON_K
        if tm > merged['moon'].max(): continue
        tr = merged[merged['moon'] <= T]
        yt = tr['target'].values.astype(np.float32)
        Xt = tr[feat].values.astype(np.float32)
        test = merged[merged['moon'] == tm]
        Xte = test[feat].values.astype(np.float32)
        y_te = test['target'].values
        t0 = time.time()
        pred = model_fn(Xt, yt, Xte, tr=tr)
        r = pearsonr(pred, y_te)[0]
        scores.append(r)
        print(f"  [{name}] T={T} tm={tm} pearson={r:+.5f} ({time.time()-t0:.0f}s)", flush=True)
    return scores


# Baseline: standard XGB
print("=== Experiment 1: BASELINE (v14 config) ===")
def bs(Xt, yt, Xte, **kw):
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt); return m.predict(Xte)
RES['1_baseline'] = k9_eval('baseline', bs)


# Experiment 2: Sample weighting — upweight non-zero rows
print("\n=== Experiment 2: Heavy sample weight non-zero rows ===")
def heavy_sw(Xt, yt, Xte, **kw):
    w = np.where(np.abs(yt) > 0.5, 10.0, 1.0).astype(np.float32)
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt, sample_weight=w); return m.predict(Xte)
RES['2_sample_weight'] = k9_eval('sample_weight', heavy_sw)


# Experiment 3: Huber loss
print("\n=== Experiment 3: Huber loss (robust to outliers) ===")
def huber(Xt, yt, Xte, **kw):
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     objective='reg:pseudohubererror',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt); return m.predict(Xte)
RES['3_huber'] = k9_eval('huber', huber)


# Experiment 4: train on RANK target then predict
print("\n=== Experiment 4: Rank target per moon ===")
def rank_target(Xt, yt, Xte, tr=None, **kw):
    # Per-moon rank of target
    tr_df = tr.copy()
    tr_df['y_rank'] = tr_df.groupby('moon')['target'].rank(pct=True)
    yt_rank = tr_df['y_rank'].values.astype(np.float32)
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt_rank); return m.predict(Xte)
RES['4_rank_target'] = k9_eval('rank_target', rank_target)


# Experiment 5: per-moon residualize — train predicting (target - moon_mean)
print("\n=== Experiment 5: Per-moon target centering ===")
def per_moon_center(Xt, yt, Xte, tr=None, **kw):
    moon_means = tr.groupby('moon')['target'].transform('mean').values.astype(np.float32)
    yt_c = yt - moon_means
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                     device=DEV, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xt, yt_c); return m.predict(Xte)
RES['5_per_moon_center'] = k9_eval('per_moon_center', per_moon_center)


# Save results
print("\n=== R8 SUMMARY ===")
for name, scores in RES.items():
    if scores:
        print(f"  {name:<25} mean={np.mean(scores):+.5f}  std={np.std(scores):.5f}  folds={[round(s,4) for s in scores]}")

# Quick claims to verify
print(f"\nClaim: sample_weight=10 helps?  diff vs baseline mean: {np.mean(RES['2_sample_weight'])-np.mean(RES['1_baseline']):+.5f}")
print(f"Claim: huber loss helps?       diff: {np.mean(RES['3_huber'])-np.mean(RES['1_baseline']):+.5f}")
print(f"Claim: rank target helps?      diff: {np.mean(RES['4_rank_target'])-np.mean(RES['1_baseline']):+.5f}")
print(f"Claim: per-moon center helps?  diff: {np.mean(RES['5_per_moon_center'])-np.mean(RES['1_baseline']):+.5f}")

with open('/workspace/results/round8_research.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in RES.items()}, f, indent=2)
