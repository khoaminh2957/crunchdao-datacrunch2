"""Round 3: measure SIGN ACCURACY of XGB predictions on the 12% non-zero target rows.

If XGB picks sign correctly for non-zero targets, we'd score much higher.
"""
import pandas as pd, numpy as np, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

# Train XGB on 1-770, test on moons 771-781
print("Training XGB 1-770, testing on moons 771-781...")
from xgboost import XGBRegressor
tr = merged[merged['moon'] <= 770]
yt = tr['target'].values.astype(np.float32)
Xt = tr[feat].values.astype(np.float32)
m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                 colsample_bytree=0.5, tree_method='hist', device='cuda:0', n_jobs=-1, random_state=42, verbosity=0)
m.fit(Xt, yt)

# Test per moon 771-781
print(f"\n{'Moon':<6} | {'n_test':<8} | {'Pearson':<8} | {'Sign acc (non-zero)':<22} | {'Mean |pred|':<12} | {'Pred quantile spread':<22}")
print('-' * 110)
for tm in range(771, 782):
    sub = merged[merged['moon'] == tm]
    if len(sub) < 100: continue
    pred = m.predict(sub[feat].values.astype(np.float32))
    target = sub['target'].values
    # Pearson
    r = pearsonr(pred, target)[0]
    # Sign accuracy for non-zero target rows
    non_zero_mask = np.abs(target) > 0.5
    if non_zero_mask.sum() > 5:
        sign_acc = (np.sign(pred[non_zero_mask]) == np.sign(target[non_zero_mask])).mean()
    else:
        sign_acc = -1
    # Pred stats
    pred_abs_mean = np.abs(pred).mean()
    q05 = np.quantile(pred, 0.05)
    q95 = np.quantile(pred, 0.95)
    print(f"{tm:<6} | {len(sub):<8} | {r:+.5f} | {sign_acc*100:>5.1f}% ({non_zero_mask.sum()} rows)   | {pred_abs_mean:.5f}      | [{q05:+.4f}, {q95:+.4f}]")

# Compute overall stats
print("\n=== Sign accuracy breakdown ===")
print("For moon 781 specifically:")
sub = merged[merged['moon'] == 781]
pred = m.predict(sub[feat].values.astype(np.float32))
target = sub['target'].values

# Group sign accuracy by prediction magnitude
print("\nSign accuracy by prediction confidence (|pred|):")
qs = np.quantile(np.abs(pred), [0.5, 0.75, 0.9, 0.95, 0.99])
for thr in qs:
    high_conf = np.abs(pred) >= thr
    if high_conf.sum() < 10: continue
    non_zero = np.abs(target[high_conf]) > 0.5
    if non_zero.sum() < 5: continue
    acc = (np.sign(pred[high_conf][non_zero]) == np.sign(target[high_conf][non_zero])).mean()
    print(f"  |pred| >= {thr:.4f} (top {(1 - np.searchsorted(np.sort(np.abs(pred)), thr) / len(pred))*100:.0f}%): "
          f"sign acc = {acc*100:.1f}% on {non_zero.sum()} non-zero rows")

# CONFUSION MATRIX
print("\nConfusion matrix on all moon 781 rows:")
pos_pred = pred > 0.05
neg_pred = pred < -0.05
zero_pred = ~(pos_pred | neg_pred)
pos_true = target > 0.5
neg_true = target < -0.5
zero_true = ~(pos_true | neg_true)

print("                  pred:-1  pred:0  pred:+1")
for tlbl, tmsk in [('true:-1', neg_true), ('true: 0', zero_true), ('true:+1', pos_true)]:
    counts = [
        (tmsk & neg_pred).sum(),
        (tmsk & zero_pred).sum(),
        (tmsk & pos_pred).sum(),
    ]
    print(f"  {tlbl:<12}  {counts[0]:<8} {counts[1]:<8} {counts[2]:<8}")
