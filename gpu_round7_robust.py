"""Round 7: ROBUST verification of 2-stage thr=0.5 lift.

Previous claim: R6 found +0.017 lift over baseline on 5 train_ends (T=600..770).
Now test on 10 train_ends + 5 seeds = 50 samples for tight confidence interval.
"""
import pandas as pd, numpy as np, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr
from xgboost import XGBRegressor, XGBClassifier

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

TRAIN_ENDS = [500, 550, 600, 625, 650, 675, 700, 725, 750, 770]
SEEDS = [42, 7, 2026]
HORIZON_K = 9

baseline_scores = []
two_stage_scores = []

for T in TRAIN_ENDS:
    tm = T + HORIZON_K
    if tm > merged['moon'].max(): continue
    tr = merged[merged['moon'] <= T]
    yt = tr['target'].values.astype(np.float32)
    Xt = tr[feat].values.astype(np.float32)
    test = merged[merged['moon'] == tm]
    Xte = test[feat].values.astype(np.float32)
    y_te = test['target'].values

    for s in SEEDS:
        # Baseline
        m_b = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
        m_b.fit(Xt, yt)
        p_b = m_b.predict(Xte)
        r_b = pearsonr(p_b, y_te)[0]
        baseline_scores.append(r_b)

        # 2-stage thr=0.5
        y_nz = (np.abs(yt) > 0.5).astype(np.int32)
        m_c = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                            device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
        m_c.fit(Xt, y_nz)
        p_c = m_c.predict_proba(Xte)[:, 1]
        mask = y_nz == 1
        m_r = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.5, tree_method='hist',
                           device='cuda:0', n_jobs=-1, random_state=s, verbosity=0)
        m_r.fit(Xt[mask], yt[mask])
        p_r = m_r.predict(Xte)
        p_2s = p_c * p_r
        r_2s = pearsonr(p_2s, y_te)[0]
        two_stage_scores.append(r_2s)

        print(f"T={T:<5} seed={s:<5} test_moon={tm}: baseline={r_b:+.5f} 2stage={r_2s:+.5f}", flush=True)

baseline_scores = np.array(baseline_scores)
two_stage_scores = np.array(two_stage_scores)
print(f"\n=== ROBUST R7 results ({len(baseline_scores)} samples = {len(TRAIN_ENDS)} train_ends × {len(SEEDS)} seeds) ===")
print(f"  Baseline mean: {baseline_scores.mean():+.5f}  std: {baseline_scores.std():.5f}")
print(f"  2-stage mean:  {two_stage_scores.mean():+.5f}  std: {two_stage_scores.std():.5f}")
print(f"  LIFT (mean):   {two_stage_scores.mean() - baseline_scores.mean():+.5f}")

# Paired t-test
diffs = two_stage_scores - baseline_scores
print(f"  Paired diff:   mean={diffs.mean():+.5f}  std={diffs.std():.5f}")
print(f"  Wins (2-stage > baseline): {(diffs > 0).sum()}/{len(diffs)} = {(diffs > 0).mean()*100:.1f}%")
print(f"  SE of mean diff: {diffs.std() / np.sqrt(len(diffs)):.5f}")
print(f"  t-stat: {diffs.mean() / (diffs.std() / np.sqrt(len(diffs))):.2f}")
