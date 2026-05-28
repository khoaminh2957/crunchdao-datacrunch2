"""Sweep on RAW target (Pearson scoring confirmed from docs).

All previous sweeps trained on rank-per-moon target. That's wrong for Pearson scoring.
This sweep uses RAW target + validation with proper 90-moon gap to simulate cloud test.
"""
import pandas as pd, numpy as np, time, json, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr


def per_moon_pearson(pred_df, truth_df):
    """Pearson correlation per moon, averaged. Matches cloud scoring."""
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    scores = []
    for m, sub in merged.groupby('moon'):
        if len(sub) < 5: continue
        # If predictions are constant -> Pearson undefined -> 0
        if sub['prediction'].std() < 1e-10:
            scores.append(0.0); continue
        if sub['target'].std() < 1e-10:
            scores.append(0.0); continue
        r = pearsonr(sub['prediction'], sub['target'])[0]
        if np.isfinite(r): scores.append(r)
    return float(np.mean(scores)) if scores else float('nan')


CONFIGS = [
    # Conservative anchors
    ('500/d6/lr=0.03 (v14)',     dict(n_estimators=500, max_depth=6, learning_rate=0.03)),
    ('500/d5/lr=0.05',           dict(n_estimators=500, max_depth=5, learning_rate=0.05)),
    ('700/d6/lr=0.05',           dict(n_estimators=700, max_depth=6, learning_rate=0.05)),
    # Moderate
    ('700/d7/lr=0.05',           dict(n_estimators=700, max_depth=7, learning_rate=0.05)),
    ('1000/d6/lr=0.03',          dict(n_estimators=1000, max_depth=6, learning_rate=0.03)),
    ('1000/d7/lr=0.03',          dict(n_estimators=1000, max_depth=7, learning_rate=0.03)),
    # Aggressive (will compare vs rank-target version)
    ('1500/d8/lr=0.04 (v52 cfg)', dict(n_estimators=1500, max_depth=8, learning_rate=0.04)),
    ('1500/d9/lr=0.07 (v51 cfg)', dict(n_estimators=1500, max_depth=9, learning_rate=0.07)),
]

# Use 90-moon-gap validation (mimics cloud)
TRAIN_ENDS = [500, 550, 600, 650]  # 4 folds
TEST_OFFSET = 90
TEST_HORIZON = 9


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    splits = []
    for T in TRAIN_ENDS:
        tr = X['moon'] <= T
        te = (X['moon'] >= T + TEST_OFFSET) & (X['moon'] < T + TEST_OFFSET + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] >= T + TEST_OFFSET) & (y['moon'] < T + TEST_OFFSET + TEST_HORIZON)].reset_index(drop=True)
        if X_te.empty: continue
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        # RAW target
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'Xt': Xt, 'yt': yt, 'X_te': X_te, 'y_te': y_te})
        print(f"  T={T} → test {T+TEST_OFFSET}..{T+TEST_OFFSET+TEST_HORIZON-1}: train={Xt.shape}, test={X_te.shape}")

    from xgboost import XGBRegressor
    print(f"\n{'Config':<30} | {'Mean Pearson':>12} | {'Std':>8} | Per-fold")
    print('-'*100)
    results = []
    for name, cfg in CONFIGS:
        fold_scores = []
        t0 = time.time()
        for s in splits:
            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
                         subsample=0.8, colsample_bytree=0.5)
            params.update(cfg)
            m = XGBRegressor(**params)
            m.fit(s['Xt'], s['yt'])
            pred_arr = m.predict(s['X_te'][feat].values.astype(np.float32))
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            sc = per_moon_pearson(pred, s['y_te'])
            fold_scores.append(sc)
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        elapsed = time.time() - t0
        results.append({'name': name, **cfg, 'mean': mean_s, 'std': std_s, 'folds': fold_scores, 'time': elapsed})
        print(f"{name:<30} | {mean_s:>+12.5f} | {std_s:>8.5f} | {[round(s, 4) for s in fold_scores]} ({elapsed:.0f}s)")
        with open('/workspace/sweep_raw.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
