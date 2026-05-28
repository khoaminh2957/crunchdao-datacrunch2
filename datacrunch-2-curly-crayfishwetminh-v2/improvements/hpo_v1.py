"""
CrunchDAO DataCrunch #2 — Bayesian HPO for LightGBM (Agent 6/10).

Search:
  - num_leaves        : [15, 31, 63, 127, 255]
  - learning_rate     : [0.01, 0.05, 0.1]
  - min_data_in_leaf  : [50, 100, 200, 500]
  - colsample_bytree  : [0.3, 0.5, 0.7, 1.0]
  - reg_lambda        : [0, 1, 5, 10]
  - n_estimators      : [200, 500, 1000]

Objective: mean per-moon Spearman correlation on holdout (last 20% of train moons).
Budget   : 50 trials, 5 min wall-clock.
Sampling : 50k rows; if `selected_features.json` exists, use its top 200 features.
"""
from __future__ import annotations
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMP_DIR = REPO_ROOT / "improvements"
IMP_DIR.mkdir(exist_ok=True)

X_PATH = DATA_DIR / "X.reduced.parquet"
Y_PATH = DATA_DIR / "y.reduced.parquet"
SELECTED_FEATURES_JSON = IMP_DIR / "selected_features.json"

BEST_PARAMS_JSON = IMP_DIR / "best_params_v1.json"
TRIALS_LOG_JSON = IMP_DIR / "hpo_v1_trials.json"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SAMPLE_ROWS = 50_000
TOP_FEATURES = 200
HOLDOUT_FRAC = 0.20
N_TRIALS = 50
TIME_BUDGET_S = 5 * 60  # 5 minutes
SEED = 42

SEARCH_SPACE = {
    "num_leaves":       [15, 31, 63, 127, 255],
    "learning_rate":    [0.01, 0.05, 0.1],
    "min_data_in_leaf": [50, 100, 200, 500],
    "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
    "reg_lambda":       [0, 1, 5, 10],
    "n_estimators":     [200, 500, 1000],
}


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
def load_data():
    print(f"[load] reading {X_PATH.name} + {Y_PATH.name} ...")
    t0 = time.time()
    X = pd.read_parquet(X_PATH)
    y = pd.read_parquet(Y_PATH)
    print(f"[load] X={X.shape}  y={y.shape}  ({time.time()-t0:.1f}s)")

    # Feature pool
    feat_cols_all = [c for c in X.columns if c not in ("id", "Id", "moon", "Moon")]

    # Optional shortlist
    if SELECTED_FEATURES_JSON.exists():
        with open(SELECTED_FEATURES_JSON) as f:
            shortlist = json.load(f)
        if isinstance(shortlist, dict) and "features" in shortlist:
            shortlist = shortlist["features"]
        shortlist = [c for c in shortlist if c in X.columns][:TOP_FEATURES]
        if len(shortlist) > 0:
            feat_cols = shortlist
            print(f"[load] using {len(feat_cols)} features from selected_features.json")
        else:
            feat_cols = feat_cols_all
            print(f"[load] selected_features.json had no matching cols → all {len(feat_cols)}")
    else:
        feat_cols = feat_cols_all
        print(f"[load] no selected_features.json → using all {len(feat_cols)} features")

    # Holdout split: last 20% moons in y
    moons = np.sort(y["moon"].unique())
    cut = moons[int(len(moons) * (1 - HOLDOUT_FRAC))]
    train_moons = moons[moons < cut]
    hold_moons = moons[moons >= cut]
    print(f"[split] train_moons={len(train_moons)} [{train_moons[0]}..{train_moons[-1]}]  "
          f"hold_moons={len(hold_moons)} [{hold_moons[0]}..{hold_moons[-1]}]")

    # Merge X+y on id,moon
    keys = [k for k in ("id", "moon") if k in X.columns and k in y.columns]
    merged = X.merge(y, on=keys, how="inner")
    merged = merged[merged["target"].notna()].reset_index(drop=True)
    print(f"[merge] merged={merged.shape}")

    train_mask = merged["moon"].isin(train_moons)
    hold_mask = merged["moon"].isin(hold_moons)

    # Sample SAMPLE_ROWS from train
    rng = np.random.default_rng(SEED)
    train_idx = np.flatnonzero(train_mask.values)
    if len(train_idx) > SAMPLE_ROWS:
        train_idx = rng.choice(train_idx, SAMPLE_ROWS, replace=False)
    train_df = merged.iloc[train_idx]

    # Holdout: keep ALL rows in holdout moons but cap for speed
    hold_idx = np.flatnonzero(hold_mask.values)
    HOLD_CAP = 100_000
    if len(hold_idx) > HOLD_CAP:
        hold_idx = rng.choice(hold_idx, HOLD_CAP, replace=False)
    hold_df = merged.iloc[hold_idx]

    print(f"[sample] train_rows={len(train_df)}  hold_rows={len(hold_df)}  "
          f"hold_moons_present={hold_df['moon'].nunique()}")

    Xt = train_df[feat_cols].astype(np.float32).values
    yt = train_df["target"].astype(np.float32).values
    Xh = hold_df[feat_cols].astype(np.float32).values
    yh = hold_df["target"].astype(np.float32).values
    mh = hold_df["moon"].astype(np.int32).values

    return Xt, yt, Xh, yh, mh, feat_cols


def per_moon_spearman(y_true: np.ndarray, y_pred: np.ndarray, moons: np.ndarray) -> float:
    """Mean per-moon Spearman correlation (ignores moons w/ <5 samples or constant y)."""
    scores = []
    for m in np.unique(moons):
        mask = moons == m
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.unique(yt).size < 2 or np.unique(yp).size < 2:
            continue
        rho, _ = spearmanr(yt, yp)
        if np.isfinite(rho):
            scores.append(rho)
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------
# Optuna objective
# ----------------------------------------------------------------------
def make_objective(Xt, yt, Xh, yh, mh):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves":       trial.suggest_categorical("num_leaves",       SEARCH_SPACE["num_leaves"]),
            "learning_rate":    trial.suggest_categorical("learning_rate",    SEARCH_SPACE["learning_rate"]),
            "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", SEARCH_SPACE["min_data_in_leaf"]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", SEARCH_SPACE["colsample_bytree"]),
            "reg_lambda":       trial.suggest_categorical("reg_lambda",       SEARCH_SPACE["reg_lambda"]),
            "n_estimators":     trial.suggest_categorical("n_estimators",     SEARCH_SPACE["n_estimators"]),
        }
        t0 = time.time()
        model = LGBMRegressor(
            n_jobs=-1,
            verbose=-1,
            random_state=SEED,
            **params,
        )
        model.fit(Xt, yt)
        preds = model.predict(Xh)
        score = per_moon_spearman(yh, preds, mh)
        dt = time.time() - t0
        print(f"  [trial {trial.number:02d}] score={score:+.5f}  params={params}  ({dt:.1f}s)")
        return score
    return objective


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    t_start = time.time()
    Xt, yt, Xh, yh, mh, feat_cols = load_data()

    print(f"\n[hpo] starting Optuna: n_trials<={N_TRIALS}, time_budget={TIME_BUDGET_S}s")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(Xt, yt, Xh, yh, mh),
        n_trials=N_TRIALS,
        timeout=TIME_BUDGET_S,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n[hpo] DONE in {time.time()-t_start:.1f}s")
    print(f"[hpo] best_score = {best.value:+.5f}")
    print(f"[hpo] best_params = {best.params}")
    print(f"[hpo] n_complete_trials = {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # Persist
    out = {
        "best_score_holdout_spearman": float(best.value),
        "best_params": best.params,
        "n_trials_run": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "sample_rows": SAMPLE_ROWS,
        "n_features_used": len(feat_cols),
        "holdout_frac": HOLDOUT_FRAC,
        "seed": SEED,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[hpo] saved → {BEST_PARAMS_JSON}")

    trials_log = [
        {"n": t.number, "score": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    with open(TRIALS_LOG_JSON, "w") as f:
        json.dump(trials_log, f, indent=2)
    print(f"[hpo] saved → {TRIALS_LOG_JSON}")


if __name__ == "__main__":
    main()
