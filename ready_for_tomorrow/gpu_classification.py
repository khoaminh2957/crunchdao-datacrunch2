"""3-class XGBClassifier on CrunchDAO target.

Binning: continuous target → 3 classes by sign
  target < -1e-6   → class 0 (negative)
  |target| <= 1e-6 → class 1 (zero, dominant ~88%)
  target >  1e-6   → class 2 (positive)

XGBClassifier(multi:softprob) → P(0), P(1), P(2)
Continuous score: P(2) - P(0)  (in [-1, 1], mimics target sign/strength)
Validate via TRUE 10-fold walk-forward (embargo=4) → Pearson(score, raw target) on LAST moon.

Memory-tight version: build each fold on demand, free immediately.
"""
import pandas as pd, numpy as np, time, json, sys, gc
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id', 'moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


def per_moon_pearson_all(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id', 'moon'], how='inner')
    return merged.groupby('moon').apply(
        lambda g: g['prediction'].corr(g['target'], method='pearson') if g['prediction'].std() > 1e-10 else 0.0,
        include_groups=False
    ).fillna(0)


def to_classes(y):
    eps = 1e-6
    cls = np.ones(len(y), dtype=np.int32)
    cls[y < -eps] = 0
    cls[y > eps] = 2
    return cls


TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
EMBARGO = 4
TEST_HORIZON = 5


CONFIGS = [
    ('cls 500/d6/lr=0.03 unweighted',  dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                                            min_child_weight=100), False),
    ('cls 500/d6/lr=0.03 balanced',    dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                                            min_child_weight=100), True),
    ('cls 300/d5/lr=0.05 unweighted',  dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                                            min_child_weight=200), False),
    ('cls 300/d5/lr=0.05 balanced',    dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                                            min_child_weight=200), True),
    ('cls 500/d5/lr=0.03 balanced',    dict(n_estimators=500, max_depth=5, learning_rate=0.03,
                                            min_child_weight=200), True),
]


def main():
    print("Loading...", flush=True)
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id', 'moon')]
    print(f"  X: {X.shape}, y: {y.shape}, features: {len(feat)}", flush=True)

    # Pre-merge whole train data once (memory: ~1.6M rows × 1150 features = ~7.5GB)
    Xy = X.merge(y, on=['id', 'moon'], how='inner')
    mask = Xy['target'].notna()
    Xy = Xy.loc[mask].reset_index(drop=True)
    print(f"  Xy merged: {Xy.shape}", flush=True)
    del X, y; gc.collect()

    from xgboost import XGBClassifier
    print(f"\n{'Config':<36} | {'MeanLast':>9} | {'StdLast':>7} | {'MeanAll':>9} | Per-fold", flush=True)
    print('-' * 130, flush=True)
    results = []
    for name, cfg, balanced in CONFIGS:
        last_scores, all_scores_per_fold = [], []
        t0 = time.time()
        for T in TRAIN_ENDS:
            tr = Xy[Xy['moon'] <= T]
            te_start = T + EMBARGO + 1
            te_end = T + EMBARGO + TEST_HORIZON
            te = Xy[(Xy['moon'] >= te_start) & (Xy['moon'] <= te_end)]
            if te.empty:
                continue
            yt_raw = tr['target'].values.astype(np.float32)
            yt_cls = to_classes(yt_raw)
            Xt = tr[feat].values.astype(np.float32)
            X_te = te[feat].values.astype(np.float32)
            y_te = te[['id', 'moon', 'target']].copy()
            last_moon = T + EMBARGO + TEST_HORIZON

            params = dict(tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                          objective='multi:softprob', num_class=3)
            params.update(cfg)
            sw = None
            if balanced:
                counts = np.bincount(yt_cls, minlength=3).astype(np.float64)
                inv = 1.0 / np.maximum(counts, 1)
                inv = inv / inv.mean()
                sw = inv[yt_cls].astype(np.float32)
            m = XGBClassifier(**params)
            m.fit(Xt, yt_cls, sample_weight=sw)
            proba = m.predict_proba(X_te)
            score = (proba[:, 2] - proba[:, 0]).astype(np.float32)
            pred = pd.DataFrame({'id': te['id'].values, 'moon': te['moon'].values,
                                 'prediction': score})
            ls = pearson_last_moon(pred, y_te)
            all_pm = per_moon_pearson_all(pred, y_te)
            last_scores.append(ls)
            all_scores_per_fold.append(float(all_pm.mean()))
            print(f"    [T={T}] last_moon={last_moon} fold_last={ls:+.5f} mean_all={all_pm.mean():+.5f}", flush=True)
            del m, Xt, X_te, yt_raw, yt_cls, proba, score, pred
            gc.collect()
        mean_last = float(np.mean(last_scores))
        std_last = float(np.std(last_scores))
        mean_all = float(np.mean(all_scores_per_fold))
        elapsed = time.time() - t0
        results.append({'name': name, 'balanced': balanced, **cfg,
                        'mean_last_moon': mean_last, 'std_last_moon': std_last,
                        'mean_all_moons': mean_all, 'folds_last': last_scores, 'time': elapsed})
        print(f"{name:<36} | {mean_last:>+9.5f} | {std_last:>7.5f} | {mean_all:>+9.5f} | "
              f"{[round(x, 4) for x in last_scores]} ({elapsed:.0f}s)", flush=True)
        with open('/workspace/sweep_classification.json', 'w') as f:
            json.dump(results, f, indent=2)
    print("\n=== Saved /workspace/sweep_classification.json ===", flush=True)
    print(f"\nBaseline v14 (XGBRegressor 500/d6/lr=0.03): MeanLast = +0.06480", flush=True)
    best = max(results, key=lambda r: r['mean_last_moon'])
    print(f"Best classifier: {best['name']} → MeanLast = {best['mean_last_moon']:+.5f}", flush=True)


if __name__ == '__main__':
    main()
