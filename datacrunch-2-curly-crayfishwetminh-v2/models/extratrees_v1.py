"""ExtraTrees regressor for CrunchDAO DataCrunch #2 — ensemble-diversity model.

Trains sklearn.ensemble.ExtraTreesRegressor on per-moon-ranked target.
ExtraTrees samples splits randomly (vs gradient boosting picking best splits),
which gives diverse predictions for stacking with the LGBM baseline (0.0124).

Saves (imputer, model, feature_cols) to model.pkl.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer


def _rank_per_moon(y_df: pd.DataFrame, moon_col: str, target_col: str) -> pd.Series:
    """Per-moon percentile rank — Spearman-native target."""
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
):
    """Train ExtraTrees on per-moon-ranked target. ExtraTrees cannot consume NaN,
    so we median-impute beforehand and persist the imputer."""
    feature_cols = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    print(f"[train] X shape={X_train.shape}, y shape={y_train.shape}, n_features={len(feature_cols)}")

    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col_raw = (
        "target" if "target" in merged.columns
        else [c for c in y_train.columns if c not in ("id", "moon")][0]
    )
    print(f"[train] target='{target_col_raw}', merged={merged.shape}")

    mask = merged[target_col_raw].notna()
    merged = merged.loc[mask].reset_index(drop=True)

    moon_col = "moon" if "moon" in merged.columns else "Moon"
    y_ranked = _rank_per_moon(
        merged[[moon_col, target_col_raw]], moon_col=moon_col, target_col=target_col_raw
    )
    print(f"[train] rank target: mean={float(y_ranked.mean()):.3f}, std={float(y_ranked.std()):.3f}")

    Xt = merged[feature_cols]
    print(f"[train] imputing NaN (median) on {Xt.shape}")
    imputer = SimpleImputer(strategy="median")
    Xt_imp = imputer.fit_transform(Xt)
    print(f"[train] imputed shape={Xt_imp.shape}")

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=200,
        max_features=0.4,
        n_jobs=-1,
        random_state=42,
    )
    print(f"[train] fitting ExtraTrees(n_est=300, depth=12, leaf=200, max_feat=0.4) on "
          f"{Xt_imp.shape[0]} rows × {Xt_imp.shape[1]} features")
    model.fit(Xt_imp, y_ranked.values)

    os.makedirs(model_directory_path, exist_ok=True)
    out_path = f"{model_directory_path}/model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((imputer, model, feature_cols), f)
    print(f"[train] saved {out_path} — {len(feature_cols)} features")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
) -> pd.DataFrame:
    """Predict for test set. Returns [id, moon, prediction]."""
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        imputer, model, feature_cols = pickle.load(f)
    print(f"[infer] X shape={X_test.shape}, cols sample={list(X_test.columns)[:5]}")

    # Tolerate missing/extra columns
    feature_cols = [c for c in feature_cols if c in X_test.columns]
    Xt = X_test[feature_cols]
    Xt_imp = imputer.transform(Xt)
    preds = model.predict(Xt_imp)

    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    out = pd.DataFrame({
        id_col: X_test[id_col].values,
        moon_col: X_test[moon_col].values,
        "prediction": preds,
    })
    print(f"[infer] returning {out.shape}, cols={list(out.columns)}")
    return out


if __name__ == "__main__":
    # Smoke test: 1000 synthetic rows × ~50 features × 5 moons
    import shutil
    import tempfile

    rng = np.random.default_rng(0)
    N = 1000
    N_FEAT = 50
    N_MOONS = 5

    ids = np.arange(N)
    moons = rng.integers(0, N_MOONS, size=N)
    X_data = rng.standard_normal((N, N_FEAT)).astype(np.float32)
    # Inject ~5% NaN to verify imputer path
    nan_mask = rng.random(X_data.shape) < 0.05
    X_data[nan_mask] = np.nan
    feat_names = [f"f{i}" for i in range(N_FEAT)]

    X_train = pd.DataFrame(X_data, columns=feat_names)
    X_train.insert(0, "id", ids)
    X_train.insert(1, "moon", moons)

    # Target = noisy linear combo of first 5 features
    signal = np.nansum(X_data[:, :5], axis=1)
    target = signal + rng.standard_normal(N) * 2.0
    y_train = pd.DataFrame({"id": ids, "moon": moons, "target": target.astype(np.float32)})

    tmpdir = tempfile.mkdtemp(prefix="extratrees_smoke_")
    try:
        print(f"[smoke] tmpdir={tmpdir}")
        train(X_train, y_train, tmpdir)
        preds = infer(X_train, tmpdir)
        assert list(preds.columns) == ["id", "moon", "prediction"], f"bad cols: {preds.columns}"
        assert len(preds) == N, f"bad len: {len(preds)}"
        assert preds["prediction"].notna().all(), "NaN in predictions"
        print(f"[smoke] OK — preds head:\n{preds.head()}")
        print(f"[smoke] pred stats: mean={preds['prediction'].mean():.4f} "
              f"std={preds['prediction'].std():.4f} "
              f"min={preds['prediction'].min():.4f} max={preds['prediction'].max():.4f}")
        # Sanity: predictions should correlate with target
        from scipy.stats import spearmanr
        rho, _ = spearmanr(preds["prediction"].values, target)
        print(f"[smoke] Spearman(pred, target) = {rho:.4f}")
        print("[smoke] PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
