"""Round 3: push capacity ceiling (depth=10, n=2000) + seed-average top configs.

R1 best 700/d7/lr=0.05 = 0.07108
R2 best 1500/d9/lr=0.07 = 0.08256 (still climbing)
R3 tests: depth 10/11, n=2000, lr fine-grain. Also 3-seed avg of top.
"""
import pandas as pd, numpy as np, json, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr


def rank_per_moon(y_df):
    grp = y_df.groupby('moon')['target']
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def per_moon_spearman(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    scores = []
    for m, sub in merged.groupby('moon'):
        if len(sub) < 5: continue
        r = spearmanr(sub['prediction'], sub['target'])[0]
        if np.isfinite(r): scores.append(r)
    return float(np.mean(scores)) if scores else float('nan')


# R3 grid
SWEEP = []
# Push ceiling: depth 10/11, n=2000
for depth in [10, 11]:
    for n_est in [1500, 2000]:
        for lr in [0.05, 0.07]:
            SWEEP.append({'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
                          'subsample': 0.8, 'colsample_bytree': 0.5,
                          'min_child_weight': 1.0, 'reg_lambda': 1.0, 'seeds': [42]})
# Seed-ensemble of R2 top configs (3 seeds, raw-avg preds)
TOP_CFGS = [
    {'n_estimators': 1500, 'max_depth': 9, 'learning_rate': 0.07},  # R2 #1
    {'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.04},  # R2 #2
    {'n_estimators': 1000, 'max_depth': 9, 'learning_rate': 0.07},  # R2 #4 (faster)
]
for cfg in TOP_CFGS:
    SWEEP.append({**cfg, 'subsample': 0.8, 'colsample_bytree': 0.5,
                  'min_child_weight': 1.0, 'reg_lambda': 1.0, 'seeds': [42, 7, 2026]})

print(f"Total configs: {len(SWEEP)} ({len([c for c in SWEEP if len(c['seeds'])==1])} single-seed + {len([c for c in SWEEP if len(c['seeds'])>1])} seed-ensemble)")

TRAIN_ENDS = [610, 625, 640, 655, 670]
TEST_HORIZON = 9


def main():
    print("Loading...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}, y={y.shape}")

    splits = []
    for T in TRAIN_ENDS:
        tr_mask = X['moon'] <= T
        te_mask = (X['moon'] > T) & (X['moon'] <= T + TEST_HORIZON)
        X_tr = X[tr_mask].reset_index(drop=True)
        X_te = X[te_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T) & (y['moon'] <= T + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_ranked = rank_per_moon(merged[['moon','target']]).values
        Xt = merged[feat].values.astype(np.float32)
        splits.append({'T': T, 'Xt': Xt, 'yt': y_ranked, 'X_te': X_te, 'y_te': y_te})
        print(f"  T={T}: Xt={Xt.shape}, X_te={X_te.shape}")

    from xgboost import XGBRegressor

    results = []
    for cfg_i, cfg in enumerate(SWEEP):
        seeds = cfg.pop('seeds') if 'seeds' in cfg else [42]
        fold_scores = []
        t_start = time.time()
        for s in splits:
            preds_all = []
            for sd in seeds:
                params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=sd, verbosity=0)
                params.update(cfg)
                m = XGBRegressor(**params)
                m.fit(s['Xt'], s['yt'])
                preds_all.append(m.predict(s['X_te'][feat].values.astype(np.float32)))
            pred_arr = np.mean(preds_all, axis=0) if len(preds_all) > 1 else preds_all[0]
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            score = per_moon_spearman(pred, s['y_te'])
            fold_scores.append(score)
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        elapsed = time.time() - t_start
        results.append({**cfg, 'seeds': seeds, 'mean_spearman': mean_s, 'std_spearman': std_s,
                        'fold_scores': fold_scores, 'elapsed_s': elapsed})
        n_seed = len(seeds)
        tag = f"x{n_seed} seeds" if n_seed > 1 else ""
        print(f"[{cfg_i+1}/{len(SWEEP)}] mean={mean_s:+.5f} std={std_s:.5f} t={elapsed:.1f}s | "
              f"n={cfg['n_estimators']} d={cfg['max_depth']} lr={cfg['learning_rate']:.3f} {tag}")
        with open('/workspace/sweep_results_r3.json', 'w') as f:
            json.dump(results, f, indent=2)

    df = pd.DataFrame(results).sort_values(['mean_spearman','std_spearman'], ascending=[False, True])
    print("\n=== TOP 10 (round 3) ===")
    print(df[['n_estimators','max_depth','learning_rate','seeds','mean_spearman','std_spearman','elapsed_s']].head(10).to_string(index=False))
    df.to_csv('/workspace/sweep_results_r3.csv', index=False)


if __name__ == '__main__':
    main()
