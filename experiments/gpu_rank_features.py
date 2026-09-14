"""B1: Per-moon rank features (Numerai-style) — convert each feature to its per-moon percentile rank.

Hypothesis: 7-level quantized features → ranked per moon gives cleaner cross-sectional signal.
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


def rank_per_moon(X_df, feat_cols, moon_col='moon'):
    """Vectorized: rank each feature within each moon, normalized [0, 1]."""
    n = X_df.groupby(moon_col)[feat_cols].transform("count")
    return (X_df.groupby(moon_col)[feat_cols].rank(method='average') / n).astype(np.float32)


def main():
    print("Loading data...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}")

    print("Computing per-moon rank features...")
    t0 = time.time()
    X_rank = rank_per_moon(X, feat).reset_index(drop=True)
    X_rank['id'] = X['id'].values
    X_rank['moon'] = X['moon'].values
    print(f"  Rank features computed in {time.time()-t0:.1f}s, shape={X_rank.shape}")

    # CONFIGS to compare: v14 baseline + rank-only + raw+rank combined
    EXPERIMENTS = [
        ('A_v14_raw_only',  X,       feat),
        ('B_rank_only',     X_rank,  feat),
        # raw + rank combined: too many feats (2300), may OOM
        # Skip for now, comment in if want
    ]

    # 10-fold walk-forward, embargo=4
    TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
    EMBARGO = 4
    TEST_HORIZON = 5

    from xgboost import XGBRegressor
    print(f"\n{'Experiment':<25} | {'MeanLast':>9} | {'StdLast':>7} | Per-fold")
    print('-'*110)

    for name, X_data, fc in EXPERIMENTS:
        last_scores = []
        t0 = time.time()
        for T in TRAIN_ENDS:
            tr = X_data['moon'] <= T
            te = (X_data['moon'] > T + EMBARGO) & (X_data['moon'] <= T + EMBARGO + TEST_HORIZON)
            X_tr = X_data[tr].reset_index(drop=True)
            X_te = X_data[te].reset_index(drop=True)
            y_tr = y[y['moon'] <= T].reset_index(drop=True)
            y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
            if X_te.empty: continue
            merged = X_tr[['id','moon']+fc].merge(y_tr, on=['id','moon'], how='inner')
            mask = merged['target'].notna()
            merged = merged.loc[mask].reset_index(drop=True)
            yt = merged['target'].values.astype(np.float32)
            Xt = merged[fc].values.astype(np.float32)

            m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.5,
                             tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
            m.fit(Xt, yt)
            pred_arr = m.predict(X_te[fc].values.astype(np.float32))
            pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': pred_arr})
            ls = pearson_last_moon(pred, y_te)
            last_scores.append(ls)

        mean_s = float(np.mean(last_scores))
        std_s = float(np.std(last_scores))
        elapsed = time.time() - t0
        print(f"{name:<25} | {mean_s:>+9.5f} | {std_s:>7.5f} | {[round(s, 4) for s in last_scores]} ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
