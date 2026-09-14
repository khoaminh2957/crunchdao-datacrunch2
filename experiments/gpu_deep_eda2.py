"""Round 2 deep EDA + multi-round verification.

Tests:
- Feature_1 × Feature_43 cross-product corr with target
- Per-bin target distribution (which feature value bins predict positive/negative target)
- Multi-CV: run TRUE validator 3 times with different random seeds, compare distributions
- Bootstrap per-moon Pearson CI
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr


def main():
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    merged = X.merge(y, on=['id','moon'], how='inner')

    # === 1. CROSS-PRODUCT CORRELATIONS ===
    print("=== 1. Cross-product corr of top 14 features with target ===")
    TOP_14 = [f"Feature_{i}" for i in [1,2,3,4,5,6,7,43,44,45,46,47,48,49]]
    cross_results = []
    for i, f1 in enumerate(TOP_14):
        for f2 in TOP_14[i+1:]:
            cross = merged[f1] * merged[f2]
            r = np.corrcoef(cross, merged['target'])[0,1]
            cross_results.append({'feats': f"{f1}×{f2}", 'corr': r})
    df_cross = pd.DataFrame(cross_results).sort_values('corr', key=lambda s: s.abs(), ascending=False)
    print(df_cross.head(10).round(5).to_string(index=False))
    print(f"\nMax |cross| corr: {df_cross['corr'].abs().max():.5f}")
    print(f"Best single feature corr (top 14): max ≈ -0.023")
    print(f"GAIN potential from crosses: {(df_cross['corr'].abs().max() - 0.023):+.5f}")

    # === 2. PER-BIN TARGET DISTRIBUTION ===
    print("\n=== 2. For Feature_1, target distribution per bin (7 bins) ===")
    for f in ['Feature_1', 'Feature_43']:
        print(f"\n  {f}:")
        bin_stats = merged.groupby(merged[f].round(2))['target'].agg(['count','mean','std']).round(4)
        print(bin_stats.to_string())

    # === 3. BOOTSTRAP PER-MOON PEARSON CI ===
    print("\n=== 3. Bootstrap: 1000 random predictions × per-moon Pearson, distribution ===")
    test_moons = [780, 781]  # latest moons
    for tm in test_moons:
        sub = merged[merged['moon'] == tm]
        if len(sub) < 100: continue
        rng = np.random.default_rng(42)
        boot_corrs = []
        for _ in range(1000):
            rand_pred = rng.standard_normal(len(sub))
            r = pearsonr(rand_pred, sub['target'].values)[0]
            boot_corrs.append(r)
        boot_corrs = np.array(boot_corrs)
        print(f"  moon={tm} (n={len(sub)}): random pred Pearson mean={boot_corrs.mean():+.4f}, std={boot_corrs.std():.4f}")
        print(f"    95% CI: [{np.quantile(boot_corrs, 0.025):+.4f}, {np.quantile(boot_corrs, 0.975):+.4f}]")
        print(f"    Score 0.08 = {(0.08 - boot_corrs.mean()) / boot_corrs.std():.2f}σ above random")

    # === 4. MULTI-RUN VERIFY: same XGB config × 5 different bootstrap subsamples ===
    print("\n=== 4. Multi-seed verify: train XGB 5x on full data with different seeds ===")
    print("Test on last moon 781")
    last_tr = merged[merged['moon'] < 781]
    last_te = merged[merged['moon'] == 781]
    yt = last_tr['target'].values.astype(np.float32)
    Xt = last_tr[feat].values.astype(np.float32)
    Xte = last_te[feat].values.astype(np.float32)
    y_te_true = last_te['target'].values

    from xgboost import XGBRegressor
    seeds = [1, 42, 100, 999, 12345]
    preds_per_seed = []
    for s in seeds:
        t0 = time.time()
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1,
                         random_state=s, verbosity=0)
        m.fit(Xt, yt)
        p = m.predict(Xte)
        preds_per_seed.append(p)
        r = pearsonr(p, y_te_true)[0]
        print(f"  seed={s}: Pearson on moon 781 = {r:+.5f}  ({time.time()-t0:.0f}s)")

    # Per-id std across seeds
    P = np.array(preds_per_seed)
    print(f"\n  Per-ID prediction std across 5 seeds: mean={P.std(axis=0).mean():.4f}")
    print(f"  Same input, different seeds → predictions diverge by this amount.")

    # === 5. WHAT IF WE PERFECTLY PREDICT POSITIVE/NEGATIVE BINARY? ===
    print("\n=== 5. Hypothetical: if we knew sign(target) with 60%/70%/80%/90% accuracy ===")
    for tm in [770, 775, 780, 781]:
        sub = merged[merged['moon'] == tm]
        target = sub['target'].values
        # Sign-only ground truth
        truth_sign = np.sign(target)
        rng = np.random.default_rng(tm)
        for acc in [0.55, 0.60, 0.70, 0.80, 0.90]:
            # Flip wrong predictions
            flip_mask = rng.random(len(target)) > acc
            pred = truth_sign.copy()
            pred[flip_mask] = -pred[flip_mask]
            r = pearsonr(pred.astype(float), target)[0]
            print(f"  moon={tm} sign-pred {acc*100:.0f}% acc: Pearson={r:+.4f}", end="")
        print()


if __name__ == '__main__':
    main()
