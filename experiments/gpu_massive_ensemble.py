"""Massive 30-model XGB ensemble validator.

30 XGB models varying n_estimators (400/500/600) × 10 seeds, depth=6, lr=0.03.
All trained on RAW target with 1150 features.
TRUE walk-forward 5-fold CV, embargo=4, score = Pearson on LAST moon of each fold.

Score reported = mean of 5 ensemble-level last-moon Pearsons + std across folds.
(NOT 5×30 = per-model; only ONE ensemble eval per fold.)

Expected: cost = 30 models × ~10–30s × 5 folds = 25–75 min on RTX 5060 Ti.
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


# === ENSEMBLE SPEC: 30 models ===
SEEDS = [42, 7, 2026, 11, 99, 5, 17, 23, 31, 53]
N_ESTIMATORS_LIST = [500, 400, 600]  # 3 × 10 = 30 models
DEPTH = 6
LR = 0.03

ENSEMBLE = []
for n_est in N_ESTIMATORS_LIST:
    for seed in SEEDS:
        ENSEMBLE.append({'n_estimators': n_est, 'max_depth': DEPTH, 'learning_rate': LR, 'seed': seed})

# === FOLDS: 5 walk-forward folds, embargo=4, test horizon 5 moons ===
TRAIN_ENDS = [500, 550, 600, 650, 700]
EMBARGO = 4
TEST_HORIZON = 5  # T+5..T+9, LAST=T+9


def main():
    print(f"[massive_ensemble] models={len(ENSEMBLE)} seeds={SEEDS} n_est_list={N_ESTIMATORS_LIST}")
    print(f"[massive_ensemble] folds={len(TRAIN_ENDS)} embargo={EMBARGO} horizon={TEST_HORIZON}")
    print("Loading data...")
    X = pd.read_parquet("/workspace/data/X.reduced.parquet")
    y = pd.read_parquet("/workspace/data/y.reduced.parquet")
    feat = [c for c in X.columns if c not in ('id', 'moon')]
    print(f"X={X.shape} y={y.shape} n_features={len(feat)}")

    # Prebuild splits
    splits = []
    for T in TRAIN_ENDS:
        tr_mask = X['moon'] <= T
        te_mask = (X['moon'] > T + EMBARGO) & (X['moon'] <= T + EMBARGO + TEST_HORIZON)
        X_tr = X[tr_mask].reset_index(drop=True)
        X_te = X[te_mask].reset_index(drop=True)
        y_tr = y[y['moon'] <= T].reset_index(drop=True)
        y_te = y[(y['moon'] > T + EMBARGO) & (y['moon'] <= T + EMBARGO + TEST_HORIZON)].reset_index(drop=True)
        if X_te.empty:
            continue
        merged = X_tr[['id', 'moon'] + feat].merge(y_tr, on=['id', 'moon'], how='inner')
        merged = merged.loc[merged['target'].notna()].reset_index(drop=True)
        yt = merged['target'].values.astype(np.float32)
        Xt = merged[feat].values.astype(np.float32)
        Xte_arr = X_te[feat].values.astype(np.float32)
        last_moon = T + EMBARGO + TEST_HORIZON
        splits.append({
            'T': T, 'last_moon': last_moon,
            'Xt': Xt, 'yt': yt,
            'X_te_meta': X_te[['id', 'moon']].copy(),
            'Xte_arr': Xte_arr,
            'y_te': y_te,
        })
        print(f"  fold T={T}: train={Xt.shape} test={Xte_arr.shape} last={last_moon}")

    from xgboost import XGBRegressor

    t_total = time.time()
    fold_last_scores = []
    fold_all_scores = []
    per_fold_per_model_last = []  # diagnostic
    per_fold_per_model_all = []

    for fi, s in enumerate(splits):
        print(f"\n=== FOLD {fi+1}/{len(splits)} T={s['T']} last_moon={s['last_moon']} ===")
        t_fold = time.time()
        sum_pred = np.zeros(s['Xte_arr'].shape[0], dtype=np.float64)
        per_model_last = []
        per_model_all = []
        for mi, cfg in enumerate(ENSEMBLE):
            t0 = time.time()
            params = dict(
                tree_method='hist', device='cuda:0', n_jobs=-1, verbosity=0,
                subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                min_child_weight=100,
                n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
                learning_rate=cfg['learning_rate'], random_state=cfg['seed'],
            )
            m = XGBRegressor(**params)
            m.fit(s['Xt'], s['yt'])
            pred_arr = m.predict(s['Xte_arr']).astype(np.float64)
            sum_pred += pred_arr
            # Per-model diagnostic
            pred_df = pd.DataFrame({
                'id': s['X_te_meta']['id'].values,
                'moon': s['X_te_meta']['moon'].values,
                'prediction': pred_arr,
            })
            ls = pearson_last_moon(pred_df, s['y_te'])
            per_model_last.append(ls)
            per_model_all.append(float(per_moon_pearson_all(pred_df, s['y_te']).mean()))
            dt = time.time() - t0
            print(f"  [m{mi+1:02d}/{len(ENSEMBLE)}] n_est={cfg['n_estimators']} seed={cfg['seed']:4d} | last={ls:+.5f} | {dt:.1f}s")
            del m, pred_arr, pred_df
            gc.collect()

        # ENSEMBLE pred = mean of raw preds
        ens_pred = sum_pred / len(ENSEMBLE)
        ens_df = pd.DataFrame({
            'id': s['X_te_meta']['id'].values,
            'moon': s['X_te_meta']['moon'].values,
            'prediction': ens_pred,
        })
        ens_last = pearson_last_moon(ens_df, s['y_te'])
        ens_all = float(per_moon_pearson_all(ens_df, s['y_te']).mean())
        fold_last_scores.append(ens_last)
        fold_all_scores.append(ens_all)
        per_fold_per_model_last.append(per_model_last)
        per_fold_per_model_all.append(per_model_all)
        dt_fold = time.time() - t_fold
        print(f"  >>> FOLD {fi+1} ensemble: last={ens_last:+.5f} all={ens_all:+.5f} | per-model last mean={np.mean(per_model_last):+.5f} std={np.std(per_model_last):.5f} | {dt_fold:.0f}s")

        # Save intermediate
        with open('/workspace/massive_ensemble_results.json', 'w') as f:
            json.dump({
                'spec': {'seeds': SEEDS, 'n_est_list': N_ESTIMATORS_LIST, 'depth': DEPTH, 'lr': LR,
                         'embargo': EMBARGO, 'horizon': TEST_HORIZON, 'train_ends': TRAIN_ENDS},
                'folds_completed': fi + 1,
                'fold_last_scores': fold_last_scores,
                'fold_all_scores': fold_all_scores,
                'per_fold_per_model_last': per_fold_per_model_last,
                'per_fold_per_model_all': per_fold_per_model_all,
            }, f, indent=2)

    mean_last = float(np.mean(fold_last_scores))
    std_last = float(np.std(fold_last_scores))
    mean_all = float(np.mean(fold_all_scores))
    print("\n" + "=" * 80)
    print(f"FINAL ENSEMBLE ({len(ENSEMBLE)} models, {len(splits)} folds)")
    print(f"  MeanLast  = {mean_last:+.5f}")
    print(f"  StdLast   = {std_last:.5f}")
    print(f"  MeanAll   = {mean_all:+.5f}")
    print(f"  Per-fold last: {[round(s,5) for s in fold_last_scores]}")
    print(f"  Wall time: {(time.time()-t_total)/60:.1f} min")
    print("=" * 80)

    with open('/workspace/massive_ensemble_results.json', 'w') as f:
        json.dump({
            'spec': {'seeds': SEEDS, 'n_est_list': N_ESTIMATORS_LIST, 'depth': DEPTH, 'lr': LR,
                     'embargo': EMBARGO, 'horizon': TEST_HORIZON, 'train_ends': TRAIN_ENDS},
            'mean_last': mean_last, 'std_last': std_last, 'mean_all': mean_all,
            'fold_last_scores': fold_last_scores,
            'fold_all_scores': fold_all_scores,
            'per_fold_per_model_last': per_fold_per_model_last,
            'per_fold_per_model_all': per_fold_per_model_all,
            'wall_minutes': (time.time() - t_total) / 60,
        }, f, indent=2)
    print("Saved /workspace/massive_ensemble_results.json")


if __name__ == '__main__':
    main()
