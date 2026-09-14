"""Scan CrunchDAO X.reduced for industry-like features.

KEY INSIGHT: each row is a unique (stock, moon) — `id` is NOT a stock-id reused
across moons. So we cannot use intra-id persistence. Instead we detect:

  (a) low CARDINALITY per moon (industries -> ~10-100 buckets)
  (b) DISCRETENESS (integer values, or {0,1} indicators)
  (c) STABLE cardinality across moons (industry taxonomies are time-invariant)
  (d) PEAKY value distribution (a few values dominate, vs continuous floats)
  (e) WITHIN-MOON variance >> CROSS-MOON drift of those discrete values
"""
import time, numpy as np, pandas as pd

PATH = r"C:\Users\Admin\Downloads\X.reduced.parquet"
t0 = time.time()
print("loading...", flush=True)
df = pd.read_parquet(PATH)
print(f"loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)

feat_cols = [c for c in df.columns if c.startswith("Feature_")]
all_moons = np.sort(df["moon"].unique())

# stratified 40 moons
idx = np.linspace(0, len(all_moons) - 1, 40).astype(int)
moons_sub = all_moons[idx]
sub = df[df["moon"].isin(moons_sub)]
print(f"subset rows={len(sub):,} moons={len(moons_sub)}", flush=True)

# ---------- (a) cardinality per moon ----------
print("cardinality per moon...", flush=True)
card_per_moon = sub.groupby("moon")[feat_cols].nunique()
mean_card  = card_per_moon.mean(axis=0)
std_card   = card_per_moon.std(axis=0)
median_card= card_per_moon.median(axis=0)

# ---------- (b) discreteness on a sample ----------
print("discreteness...", flush=True)
samp = sub.sample(min(300000, len(sub)), random_state=0)[feat_cols]
diff = (samp - samp.round()).abs()
frac_int     = (diff < 1e-4).mean(axis=0)
frac_zero_one= ((samp.round() == 0) | (samp.round() == 1)).mean(axis=0)

# ---------- (d) peakiness: top-1 value frequency, entropy ----------
print("peakiness / top-1 freq...", flush=True)
top1_freq = {}
entropy  = {}
# round to 4 decimals to bucket near-equal floats
rounded = samp.round(4)
for c in feat_cols:
    vc = rounded[c].value_counts(normalize=True)
    if len(vc) == 0:
        top1_freq[c] = np.nan; entropy[c] = np.nan; continue
    top1_freq[c] = vc.iloc[0]
    p = vc.values
    entropy[c]  = -(p * np.log(p + 1e-12)).sum()
top1_freq = pd.Series(top1_freq)
entropy   = pd.Series(entropy)

# ---------- (c) stability of cardinality across moons ----------
# coefficient of variation of cardinality across moons. industries -> low CV
cv_card = (std_card / mean_card.replace(0, np.nan)).fillna(1.0)

# ---------- combine ----------
score = pd.DataFrame({
    "mean_card_per_moon": mean_card,
    "median_card_per_moon": median_card,
    "cv_card_across_moons": cv_card,
    "frac_integer": frac_int,
    "frac_0_or_1": frac_zero_one,
    "top1_value_freq": top1_freq,
    "entropy_nats": entropy,
})

# normalize each piece into 0..1 "industry-likeness"
def squash(s, lo, hi, invert=False):
    x = ((s - lo) / (hi - lo)).clip(0, 1)
    return 1 - x if invert else x

score["s_lowcard"]  = squash(score["mean_card_per_moon"], 2, 200, invert=True)
score["s_stable"]   = squash(score["cv_card_across_moons"], 0, 0.5, invert=True)
score["s_integer"]  = score["frac_integer"]
score["s_peaky"]    = score["top1_value_freq"]
score["s_lowent"]   = squash(score["entropy_nats"], 0, 6, invert=True)

score["industry_score"] = (
    1.5 * score["s_lowcard"]
    + 1.0 * score["s_stable"]
    + 1.0 * score["s_integer"]
    + 1.0 * score["s_peaky"]
    + 0.5 * score["s_lowent"]
)
score = score.sort_values("industry_score", ascending=False)

print("\n=== TOP 25 industry-like features ===", flush=True)
cols = ["mean_card_per_moon","cv_card_across_moons","frac_integer","frac_0_or_1","top1_value_freq","entropy_nats","industry_score"]
print(score[cols].head(25).round(3).to_string(), flush=True)

print("\n=== BOTTOM 10 (most continuous / least industry-like) ===", flush=True)
print(score[cols].tail(10).round(3).to_string(), flush=True)

# how many features have mean_card <= 50?
mask_lowcard = score["mean_card_per_moon"] <= 50
print(f"\nfeatures with mean_card_per_moon <= 50 : {mask_lowcard.sum()}", flush=True)
mask_int = score["frac_integer"] > 0.95
print(f"features almost fully integer-valued     : {mask_int.sum()}", flush=True)
mask_binary = (score["frac_0_or_1"] > 0.95)
print(f"features almost binary 0/1               : {mask_binary.sum()}", flush=True)

score.to_csv("industry_scores.csv")
print("\nsaved industry_scores.csv", flush=True)
print(f"done in {time.time()-t0:.1f}s", flush=True)
