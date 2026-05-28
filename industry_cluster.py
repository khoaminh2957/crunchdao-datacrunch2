"""All 1150 features are quantile-binned into the SAME 7-bucket marginal per moon.
So univariate signals can't separate industry from non-industry. We must use
cross-feature structure: features that encode the same industry will be highly
correlated WITHIN a moon. Industry features also tend to be persistent in their
relationships across moons (a Tech cluster stays a Tech cluster).

Approach
--------
1. compute the 1150x1150 abs-correlation matrix within each of ~10 moons
2. average them -> stable cross-moon co-movement matrix C
3. for each feature, "industry-likeness" = mean of top-K (K=10) abs correlations
   with OTHER features. Industry dummies form tight clusters of redundant
   columns (cf. F1..F2 = 0.996); idiosyncratic features have weak top-K.
4. also report cluster size at threshold 0.8.
"""
import time, numpy as np, pandas as pd

PATH = r"C:\Users\Admin\Downloads\X.reduced.parquet"
t0 = time.time()
print("loading...", flush=True)
df = pd.read_parquet(PATH)
print(f"loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)
feat_cols = [c for c in df.columns if c.startswith("Feature_")]
F = len(feat_cols)

all_moons = np.sort(df["moon"].unique())
sel_moons = all_moons[np.linspace(0, len(all_moons)-1, 10).astype(int)]
print("moons for corr:", sel_moons.tolist(), flush=True)

acc = np.zeros((F, F), dtype=np.float64)
n_used = 0
for m in sel_moons:
    block = df[df["moon"] == m][feat_cols].to_numpy(dtype=np.float32)
    # standardize
    block = block - block.mean(axis=0, keepdims=True)
    s = block.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    block = block / s
    n = block.shape[0]
    c = (block.T @ block) / max(n - 1, 1)
    acc += np.abs(c)
    n_used += 1
    print(f"  moon {m}: rows={n}", flush=True)

C = acc / n_used
np.fill_diagonal(C, 0.0)

# per-feature: mean of top-10 abs correlations
K = 10
topk_mean = np.sort(C, axis=1)[:, -K:].mean(axis=1)
# also: cluster size at threshold 0.8 and 0.5
cluster_size_08 = (C > 0.8).sum(axis=1)
cluster_size_05 = (C > 0.5).sum(axis=1)
max_corr        = C.max(axis=1)

out = pd.DataFrame({
    "feature": feat_cols,
    "top10_mean_abs_corr": topk_mean,
    "max_abs_corr_other_feat": max_corr,
    "cluster_size_thr_0.8": cluster_size_08,
    "cluster_size_thr_0.5": cluster_size_05,
})
out["industry_score"] = (
    0.6 * out["top10_mean_abs_corr"]
    + 0.4 * np.clip(out["cluster_size_thr_0.8"] / 50.0, 0, 1)
)
out = out.sort_values("industry_score", ascending=False)

print("\n=== TOP 25 industry-like features (high redundancy = likely industry dummies) ===", flush=True)
print(out.head(25).to_string(index=False), flush=True)
print("\n=== BOTTOM 10 (most idiosyncratic / stock-specific) ===", flush=True)
print(out.tail(10).to_string(index=False), flush=True)

# distribution of cluster sizes
print("\ndistribution of cluster_size_thr_0.8:", flush=True)
print(out["cluster_size_thr_0.8"].describe(), flush=True)
print("# features with cluster_size>=20 at thr0.8:",
      (out["cluster_size_thr_0.8"] >= 20).sum(), flush=True)
print("# features with cluster_size>=5  at thr0.8:",
      (out["cluster_size_thr_0.8"] >= 5).sum(), flush=True)
print("# features with top10_mean_abs_corr>=0.5:",
      (out["top10_mean_abs_corr"] >= 0.5).sum(), flush=True)

out.to_csv(r"C:\Users\Admin\earn5usd\crunchdao\industry_cluster_scores.csv", index=False)
print(f"\nsaved industry_cluster_scores.csv; done in {time.time()-t0:.1f}s", flush=True)
