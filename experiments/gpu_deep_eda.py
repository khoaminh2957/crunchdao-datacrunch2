"""Deep EDA on why scores keep dropping/varying.

Tests:
1. Per-moon target characteristics — is moon 790 unusual?
2. Per-moon feature distribution drift over time
3. Per-moon "intrinsic predictability" — how much signal exists per moon
4. Variance decomposition: train signal vs cloud test signal
5. Last-moon vs penultimate-moon comparison
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr, spearmanr


def main():
    print("Loading...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id','moon')]
    print(f"X={X.shape}, y={y.shape}, moons {X['moon'].min()}..{X['moon'].max()}")

    merged = X.merge(y, on=['id','moon'], how='inner')
    print(f"merged={merged.shape}")

    # === 1. PER-MOON TARGET DISTRIBUTION OVER TIME ===
    print("\n=== 1. Per-moon target characteristics ===")
    moon_stats = merged.groupby('moon').agg(
        n=('target','size'),
        mean_t=('target','mean'),
        std_t=('target','std'),
        zero_pct=('target', lambda x: (x==0).mean()),
        pos_pct=('target', lambda x: (x>0.5).mean()),
        neg_pct=('target', lambda x: (x<-0.5).mean()),
    ).round(4)
    # Bin by 100-moon chunks
    moon_stats['bin'] = (moon_stats.index - 1) // 100
    bin_summary = moon_stats.groupby('bin').agg(
        n=('n','mean'),
        std_t=('std_t','mean'),
        zero_pct=('zero_pct','mean'),
        pos_pct=('pos_pct','mean'),
    ).round(4)
    print("Per 100-moon bin:")
    print(bin_summary.to_string())
    print("\nLast 15 moons (where cloud tests):")
    print(moon_stats.tail(15).drop(columns='bin').to_string())

    # === 2. FEATURE DISTRIBUTION DRIFT (top features) ===
    print("\n=== 2. Top features drift over time ===")
    # For top 5 features (by EDA), compute mean per moon over time
    top_feats = ['Feature_1', 'Feature_43', 'Feature_2', 'Feature_44', 'Feature_3']
    for f in top_feats:
        if f in merged.columns:
            per_moon = merged.groupby('moon')[f].mean()
            first_100 = per_moon.iloc[:100].mean()
            last_100 = per_moon.iloc[-100:].mean()
            print(f"  {f}: first-100-moon mean={first_100:.4f}, last-100-moon mean={last_100:.4f}, diff={last_100-first_100:+.4f}")

    # === 3. PER-MOON INTRINSIC PREDICTABILITY ===
    print("\n=== 3. Per-moon Pearson(top feature, target) — how much signal per moon? ===")
    # For Feature_1, compute Pearson with target PER MOON, then look at distribution over moons
    per_moon_corr = merged.groupby('moon').apply(
        lambda g: g['Feature_1'].corr(g['target'], method='pearson'),
        include_groups=False
    )
    print(f"  Feature_1 per-moon Pearson with target:")
    print(f"    mean={per_moon_corr.mean():.4f}, std={per_moon_corr.std():.4f}")
    print(f"    min={per_moon_corr.min():.4f}, max={per_moon_corr.max():.4f}")
    print(f"    Histogram (10 bins): {np.histogram(per_moon_corr, bins=10)[0].tolist()}")

    # === 4. PREDICTABILITY BY ERA ===
    print("\n=== 4. Predictability by 100-moon era ===")
    # Per-moon Pearson with target then average within 100-moon bins
    per_moon_corr.index.name = 'moon'
    per_moon_corr_df = per_moon_corr.to_frame('corr')
    per_moon_corr_df['era'] = (per_moon_corr_df.index - 1) // 100
    era_corr = per_moon_corr_df.groupby('era').agg(
        mean_abs_corr=('corr', lambda x: x.abs().mean()),
        mean_corr=('corr', 'mean'),
        std_corr=('corr', 'std'),
    )
    print(era_corr.round(4).to_string())

    # === 5. RANDOM-BASELINE BOOTSTRAP ===
    print("\n=== 5. Random baseline: shuffle predictions, score per moon ===")
    # For each moon, what's the Pearson of RANDOM predictions with target? Should be ~0 with high std.
    np.random.seed(42)
    sample_moons = np.linspace(merged['moon'].min(), merged['moon'].max(), 30).astype(int)
    random_corrs = []
    for m in sample_moons:
        sub = merged[merged['moon'] == m]
        if len(sub) < 100: continue
        random_pred = np.random.randn(len(sub))
        r = pearsonr(random_pred, sub['target'].values)[0]
        random_corrs.append(r)
    print(f"  Random-pred per-moon Pearson: mean={np.mean(random_corrs):+.4f}, std={np.std(random_corrs):.4f}")
    print(f"  This is the NOISE floor: max possible improvement = mean ABS")

    print("\n=== 6. Theoretical ceiling: BEST POSSIBLE Pearson on test moon ===")
    # If we knew the targets ranking exactly, what's the max Pearson with raw target?
    # Pearson is scale-invariant so perfect predictor of target gives Pearson = 1.0
    # But due to the 88% zero structure, even good predictors get hurt by ties
    for m in sample_moons[-5:]:
        sub = merged[merged['moon'] == m]
        # Perfect predictor
        perfect_r = pearsonr(sub['target'].values, sub['target'].values)[0]
        # +1 noise predictor
        noisy = sub['target'].values + np.random.randn(len(sub)) * 0.5
        noisy_r = pearsonr(noisy, sub['target'].values)[0]
        # Sign predictor (only -1/0/+1)
        sign = np.sign(sub['target'].values)
        sign_r = pearsonr(sign, sub['target'].values)[0]
        print(f"  moon={m}: perfect={perfect_r:.4f}, sign-only={sign_r:.4f}, target+0.5σ-noise={noisy_r:.4f}")


if __name__ == '__main__':
    main()
