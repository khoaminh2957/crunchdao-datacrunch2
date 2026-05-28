"""B3: 3-class XGBClassifier. Predict P(class). Use P(positive) - P(negative) as score."""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')


def pearson_last_moon(pred_df, truth_df):
    merged = pred_df.merge(truth_df, on=['id','moon'], how='inner')
    last_moon = merged['moon'].max()
    sub = merged[merged['moon'] == last_moon]
    if len(sub) < 5 or sub['prediction'].std() < 1e-10 or sub['target'].std() < 1e-10:
        return 0.0
    return float(sub['prediction'].corr(sub['target'], method='pearson'))


def target_to_class(t):
    """Map target to 3-class. -1→0, 0→1, +1→2 (binarize at 0.5)."""
    out = np.full(len(t), 1, dtype=np.int32)
    out[t < -0.5] = 0
    out[t > 0.5] = 2
    return out


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]

    TRAIN_ENDS = [400, 450, 500, 525, 550, 575, 600, 625, 650, 670]
    EMBARGO = 4
    TEST_HORIZON = 5

    from xgboost import XGBClassifier

    scores = []
    t_g = time.time()
    for fi, T in enumerate(TRAIN_ENDS):
        t0 = time.time()
        tr = X['moon'] <= T
        te = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr].reset_index(drop=True)
        X_te = X[te].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        merged = X_tr[['id','moon']+feat].merge(y_tr, on=['id','moon'], how='inner')
        mask = merged['target'].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        y_class = target_to_class(merged['target'].values)
        Xt = merged[feat].values.astype(np.float32)

        m = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5,
            tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0,
            objective='multi:softprob', num_class=3,
        )
        m.fit(Xt, y_class)
        proba = m.predict_proba(X_te[feat].values.astype(np.float32))
        # Continuous score: P(class 2) - P(class 0) = P(positive) - P(negative)
        p_score = proba[:, 2] - proba[:, 0]
        pred = pd.DataFrame({'id': X_te['id'].values, 'moon': X_te['moon'].values, 'prediction': p_score})
        sc = pearson_last_moon(pred, y_te)
        scores.append(sc)
        print(f"  Fold {fi+1}/10 T={T} ({time.time()-t0:.0f}s): pearson_last_moon={sc:.5f}")

    print(f"\n=== B3 Classification ({time.time()-t_g:.0f}s) ===")
    print(f"  MeanLast={np.mean(scores):+.5f} StdLast={np.std(scores):.5f}")
    print(f"  Folds: {[round(s,4) for s in scores]}")


if __name__ == '__main__':
    main()
