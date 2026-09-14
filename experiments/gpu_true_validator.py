"""TRUE local validator mimicking REAL CrunchDAO DataCrunch #2 scoring.

Per A1 finding: score = Pearson(prediction, target) on the LAST MOON only.
Std of single-moon Pearson on 2000 stocks ~ 0.022. Need MANY folds to get reliable signal.

Strategy:
- Many fold (10-15 folds) for noise reduction
- Each fold: train on 1..T, test on T+4..T+9 (embargo 4 + 5 test moons)
- Score = Pearson on LAST test moon only (T+9)
- Mean across folds + std = reliable estimate

Configs to validate (SHALLOW per A5 'simplicity wins' research):
"""
import pandas as pd, numpy as np, time, json, sys
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    """The REAL CrunchDAO scoring: Pearson on the LAST moon of test horizon."""
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


def per_moon_pearson_all(pred_df, truth_df):
    """All per-moon Pearsons for diagnostic — see variance."""
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    return merged.groupby('moon').apply(
        lambda g: g['prediction'].corr(g['target'], method='pearson') if g['prediction'].std() > 1e-10 else 0.0,
        include_groups=False
    ).fillna(0)


# SHALLOW configs (per A5 research) + a few moderate
CONFIGS = [
    ('200/d3/lr=0.05',           dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=500)),
    ('300/d4/lr=0.05',           dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=500)),
    ('300/d5/lr=0.05',           dict(n_estimators=300, max_depth=5, learning_rate=0.05, min_child_weight=200)),
    ('500/d5/lr=0.03',           dict(n_estimators=500, max_depth=5, learning_rate=0.03, min_child_weight=200)),
    ('500/d6/lr=0.03 (v14)',     dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=100)),
    ('1000/d4/lr=0.02',          dict(n_estimators=1000, max_depth=4, learning_rate=0.02, min_child_weight=500)),
    # Reference: deep (expected to score badly here)
    ('1500/d8/lr=0.04 (v52)',    dict(n_estimators=1500, max_depth=8, learning_rate=0.04, min_child_weight=1)),
]

# 10 folds — TRAIN ends at various points, test horizon 5 moons after embargo 4
TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
EMBARGO = 4
TEST_HORIZON = 5  # T+4..T+8 (5 moons, scoring on T+8 only)


def main():
    print("Loading...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    splits = []
    for T in TRAIN_ENDS:
        tr_mask = X['moon'] <= T
        te_mask = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr_mask].reset_index(drop=True)
        X_te = X[te_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        if X_te.empty: continue
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        # USE RAW TARGET (Pearson)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        last_moon = T + EMBARGO + TEST_HORIZON
        splits.append({'T': T, 'last_moon': last_moon, 'Xt': Xt, 'yt': yt, 'X_te': X_te, 'y_te': y_te})
        print(f"  T={T} → embargo {T+1}..{T+EMBARGO} → test {T+EMBARGO+1}..{last_moon}, LAST={last_moon}: train={Xt.shape}")

    from xgboost import XGBRegressor
    print(f"\n{'Config':<28} | {'MeanLast':>9} | {'StdLast':>7} | {'MeanAll':>9} | Per-fold last-moon")
    print('-'*120)
    results = []
    for name, cfg in CONFIGS:
        last_scores = []
        all_scores_per_fold = []
        t0 = time.time()
        for s in splits:
            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0)
            params.update(cfg)
            m = XGBRegressor(**params)
            m.fit(s['Xt'], s['yt'])
            pred_arr = m.predict(s['X_te'][feat].values.astype(np.float32))
            pred = pd.DataFrame({'id': s['X_te']['id'].values, 'moon': s['X_te']['moon'].values, 'prediction': pred_arr})
            ls = pearson_last_moon(pred, s['y_te'])
            all_pm = per_moon_pearson_all(pred, s['y_te'])
            last_scores.append(ls)
            all_scores_per_fold.append(float(all_pm.mean()))
        mean_last = float(np.mean(last_scores))
        std_last = float(np.std(last_scores))
        mean_all = float(np.mean(all_scores_per_fold))
        elapsed = time.time() - t0
        results.append({'name': name, **cfg, 'mean_last_moon': mean_last, 'std_last_moon': std_last,
                        'mean_all_moons': mean_all, 'folds_last': last_scores, 'time': elapsed})
        print(f"{name:<28} | {mean_last:>+9.5f} | {std_last:>7.5f} | {mean_all:>+9.5f} | {[round(s, 4) for s in last_scores]} ({elapsed:.0f}s)")
        with open('/workspace/sweep_true.json', 'w') as f:
            json.dump(results, f, indent=2)
    print("\n=== Saved /workspace/sweep_true.json ===")


if __name__ == '__main__':
    main()
