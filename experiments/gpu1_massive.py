"""Launch on cuda:1 — massive parallel experiments.

Loop training many models, exploring different approaches.
"""
import pandas as pd, numpy as np, time, sys, json
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]

TRAIN_ENDS = [500, 550, 600, 650, 670]  # 5 folds
EMBARGO = 4
TEST_HORIZON = 5

# Precompute splits
splits = []
for T in TRAIN_ENDS:
    tr = X['moon'] <= T
    te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
    X_tr = X[tr].reset_index(drop=True)
    X_te = X[te].reset_index(drop=True)
    y_tr = y[y['moon'] <= T].reset_index(drop=True)
    y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
    merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
    mask = merged['target'].notna(); merged = merged.loc[mask].reset_index(drop=True)
    yt = merged['target'].values.astype(np.float32)
    Xt = merged[feat].values.astype(np.float32)
    Xte = X_te[feat].values.astype(np.float32)
    splits.append({'T': T, 'Xt': Xt, 'yt': yt, 'X_te': X_te, 'Xte_arr': Xte, 'y_te': y_te})
print(f"Loaded {len(splits)} splits.")

# CONFIGS to explore — varied hyperparams
CONFIGS = [
    {'name': 'd=4 mcw=200', 'params': dict(n_estimators=500, max_depth=4, learning_rate=0.03, min_child_weight=200)},
    {'name': 'd=4 mcw=500', 'params': dict(n_estimators=500, max_depth=4, learning_rate=0.03, min_child_weight=500)},
    {'name': 'd=4 mcw=1000', 'params': dict(n_estimators=500, max_depth=4, learning_rate=0.03, min_child_weight=1000)},
    {'name': 'd=5 mcw=500', 'params': dict(n_estimators=500, max_depth=5, learning_rate=0.03, min_child_weight=500)},
    {'name': 'd=5 mcw=1000', 'params': dict(n_estimators=500, max_depth=5, learning_rate=0.03, min_child_weight=1000)},
    {'name': 'd=6 mcw=200', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=200)},
    {'name': 'd=6 reg=10', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, reg_lambda=10.0)},
    {'name': 'd=6 reg=50', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, reg_lambda=50.0)},
    {'name': 'd=6 col=0.2', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, colsample_bytree=0.2)},
    {'name': 'd=6 col=0.8', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, colsample_bytree=0.8)},
    {'name': 'd=6 sub=0.5', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.5)},
    {'name': 'd=6 alpha=1', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, reg_alpha=1.0)},
    {'name': 'd=6 alpha=5', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, reg_alpha=5.0)},
    {'name': 'd=6 huber', 'params': dict(n_estimators=500, max_depth=6, learning_rate=0.03, objective='reg:pseudohubererror')},
]

from xgboost import XGBRegressor

print(f"{'Config':<30} | {'Mean':>9} | {'Std':>7} | Folds")
print('-'*100)
results = []
for cfg_i, cfg in enumerate(CONFIGS):
    fold_scores = []
    t0 = time.time()
    for s in splits:
        params = dict(tree_method='hist', device='cuda:1', n_jobs=-1, random_state=42, verbosity=0,
                      subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=1.0)
        params.update(cfg['params'])
        m = XGBRegressor(**params)
        m.fit(s['Xt'], s['yt'])
        p = m.predict(s['Xte_arr'])
        pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': p})
        sc = pearson_last_moon(pred, s['y_te'])
        fold_scores.append(sc)
    mean_s = float(np.mean(fold_scores))
    std_s = float(np.std(fold_scores))
    elapsed = time.time() - t0
    results.append({'name': cfg['name'], **cfg['params'], 'mean': mean_s, 'std': std_s, 'folds': fold_scores, 'time': elapsed})
    print(f"{cfg['name']:<30} | {mean_s:>+9.5f} | {std_s:>7.5f} | {[round(s,4) for s in fold_scores]} ({elapsed:.0f}s)")
    with open('/workspace/results/gpu1_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

# Sort + print top
df = pd.DataFrame(results).sort_values('mean', ascending=False)
print("\n=== TOP 5 ===")
print(df[['name','mean','std','time']].head(5).to_string(index=False))
