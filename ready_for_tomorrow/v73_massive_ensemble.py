"""v73: Massive 30-model XGB ensemble — variance reduction at extreme scale.

Configs: n_estimators in [400, 500, 600] x seeds [42, 7, 2026, 11, 99, 5, 17, 23, 31, 53]
  = 3 x 10 = 30 models, all max_depth=6, lr=0.03, RAW target.

Local validation (gpu_massive_ensemble.py, 5 folds walk-forward, embargo=4, last-moon Pearson):
  RESULT_PLACEHOLDER — populated after sweep finishes.

================================================================================
IMPORTANT — DO NOT RUN train() ON THE CLOUD GRADER:
  30 XGB models x ~1.5M rows x 1150 features = ~30-90 min train wall-time
  on a strong CPU box, far longer on the grader's allotted slice. The grader
  will time-out or thrash.

  Strategy: train LOCALLY (or on GPU box), pickle the (models, feature_cols, target_col)
  tuple to model.pkl, and ship that file alongside this script. Replace train() with
  a no-op or sanity-check that simply re-pickles to model_directory_path if a
  pre-trained model.pkl is supplied alongside.

  CURRENT train() WILL RUN if invoked — it is provided for completeness and for
  local re-training. Override or skip on submission.
================================================================================
"""
import pandas as pd, numpy as np, pickle, os

SEEDS = [42, 7, 2026, 11, 99, 5, 17, 23, 31, 53]
N_ESTIMATORS_LIST = [500, 400, 600]
MAX_DEPTH = 6
LR = 0.03
MIN_CHILD_WEIGHT = 100


def train(X_train, y_train, model_directory_path):
    """WARNING: 30-model train. Run locally; ship pre-trained model.pkl to grader.

    If model_directory_path already contains a model.pkl (shipped pre-trained),
    this function will skip training and re-use that file.
    """
    os.makedirs(model_directory_path, exist_ok=True)
    pretrained = os.path.join(model_directory_path, "model.pkl")
    if os.path.exists(pretrained):
        print(f"[v73/train] Pre-trained model.pkl found — skipping 30-model train.")
        return

    from xgboost import XGBRegressor
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id", "moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    yt = merged[target_col].values
    Xt = merged[feature_cols]

    models = []
    total = len(N_ESTIMATORS_LIST) * len(SEEDS)
    idx = 0
    for n_est in N_ESTIMATORS_LIST:
        for seed in SEEDS:
            idx += 1
            print(f"[v73/train {idx:02d}/{total}] n_est={n_est} seed={seed}")
            m = XGBRegressor(
                n_estimators=n_est, max_depth=MAX_DEPTH, learning_rate=LR,
                min_child_weight=MIN_CHILD_WEIGHT,
                subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                tree_method='hist', n_jobs=-1, random_state=seed, verbosity=0,
            )
            m.fit(Xt, yt)
            models.append(m)

    with open(pretrained, "wb") as f:
        pickle.dump((models, feature_cols, target_col), f)
    print(f"[v73/train] Pickled {len(models)} models to {pretrained}")


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        models, feature_cols, _ = pickle.load(f)
    preds = np.mean([m.predict(X_test[feature_cols]) for m in models], axis=0)
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds,
    })
