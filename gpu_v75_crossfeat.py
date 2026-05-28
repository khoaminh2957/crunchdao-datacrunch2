"""v75: XGB on raw + cross-product features of top 14 informative features.

Top 14 by EDA: Feature_1..7 + Feature_43..49. Add 14*14 = 196 pairwise products.
Total features: 1150 + 196 = 1346.

Hypothesis: non-linear interaction in top features captures signal XGB missed.
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


TOP_14 = [f"Feature_{i}" for i in [1,2,3,4,5,6,7,43,44,45,46,47,48,49]]


def add_cross_features(X_df):
    """Add 14*14 = 196 cross-product features."""
    out = X_df.copy()
    for f1 in TOP_14:
        for f2 in TOP_14:
            if f1 == f2: continue
            out[f"{f1}_x_{f2}"] = X_df[f1] * X_df[f2]
    return out


def main():
    print("Loading...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat_raw = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}")

    print("Building cross features (14*14 pairs)...")
    t0 = time.time()
    X_aug = add_cross_features(X)
    cross_cols = [c for c in X_aug.columns if '_x_' in c]
    feat_all = feat_raw + cross_cols
    print(f"  Added {len(cross_cols)} cross features in {time.time()-t0:.1f}s. Total: {len(feat_all)}")

    TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
    EMBARGO = 4
    TEST_HORIZON = 5

    from xgboost import XGBRegressor
    last_scores = []
    last_scores_raw = []
    t_global = time.time()
    for fi, T in enumerate(TRAIN_ENDS):
        t0 = time.time()
        tr = X_aug['moon'] <= T
        te = (X_aug['moon'] > T + EMBARGO) & (X_aug['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X_aug[tr].reset_index(drop=True)
        X_te = X_aug[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat_all].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna(); merged = merged.loc[mask].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat_all].values.astype(np.float32)

        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m.fit(Xt, yt)
        pred_arr = m.predict(X_te[feat_all].values.astype(np.float32))
        pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': pred_arr})
        sc = pearson_last_moon(pred, y_te)
        last_scores.append(sc)

        # Also baseline raw for comparison
        m2 = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
        m2.fit(merged[feat_raw].values.astype(np.float32), yt)
        pred2 = m2.predict(X_te[feat_raw].values.astype(np.float32))
        pred2_df = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': pred2})
        sc2 = pearson_last_moon(pred2_df, y_te)
        last_scores_raw.append(sc2)
        print(f"  Fold {fi+1}/10 T={T} ({time.time()-t0:.0f}s): raw={sc2:.4f} +cross={sc:.4f}")

    print(f"\n=== v75 results ({time.time()-t_global:.0f}s) ===")
    print(f"  raw v14         mean={np.mean(last_scores_raw):+.5f} std={np.std(last_scores_raw):.5f}")
    print(f"  v14 + 196 cross mean={np.mean(last_scores):+.5f} std={np.std(last_scores):.5f}")
    print(f"  LIFT: {np.mean(last_scores) - np.mean(last_scores_raw):+.5f}")


if __name__ == '__main__':
    main()
