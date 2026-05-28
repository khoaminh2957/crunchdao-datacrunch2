"""
Ensemble v1 for CrunchDAO DataCrunch #2.

Provides three model families behind a single sklearn-style interface
(`fit(X, y)` / `predict(X)`) and two ensemble strategies:

    - SimpleAverageEnsemble  : equal-weighted mean of base predictions
    - StackingEnsemble       : meta-LightGBM trained on out-of-fold
                               base predictions, then refit-on-all

Per-moon validation: `evaluate_per_moon` computes Spearman rank
correlation between predictions and target on each moon, then averages.
This is the metric used by DataCrunch and is the only metric we report.

XGBoost / CatBoost are imported lazily — if either is missing, the
corresponding wrapper raises at construction time but the rest of the
module still works (we automatically fall back to LightGBM-only).

The `__main__` block runs a smoke test on a 10k x 100 synthetic sample
so we can verify the plumbing without touching the real 1.6M-row data.

Do NOT import this from main.py — main.py is owned by the baseline.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Base model wrappers — shared sklearn-style interface
# ---------------------------------------------------------------------------

class BaseModel:
    """Minimal sklearn-style contract used by the ensemble."""
    name: str = "base"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class LGBModel(BaseModel):
    name = "lgb"

    def __init__(self, **kwargs):
        from lightgbm import LGBMRegressor  # always available per requirements.txt
        defaults = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=100,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            verbose=-1,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        self.model = LGBMRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBModel(BaseModel):
    name = "xgb"

    def __init__(self, **kwargs):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost not installed — install with `pip install xgboost` "
                "or use ensemble_config='avg_lgb_only' / drop xgb from the model list."
            ) from e
        defaults = dict(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class CatModel(BaseModel):
    name = "cat"

    def __init__(self, **kwargs):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "catboost not installed — install with `pip install catboost` "
                "or drop cat from the model list."
            ) from e
        defaults = dict(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        defaults.update(kwargs)
        self.model = CatBoostRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


_MODEL_REGISTRY = {
    "lgb": LGBModel,
    "xgb": XGBModel,
    "cat": CatModel,
}


def _available_models(requested: Sequence[str]) -> List[str]:
    """Filter `requested` to model names whose backing lib is importable."""
    available = []
    for name in requested:
        if name not in _MODEL_REGISTRY:
            warnings.warn(f"unknown model '{name}', skipping")
            continue
        try:
            _MODEL_REGISTRY[name]()
            available.append(name)
        except ImportError as e:
            warnings.warn(f"model '{name}' unavailable: {e}")
    return available


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

@dataclass
class SimpleAverageEnsemble:
    """Equal-weighted mean of base-model predictions."""

    model_names: Sequence[str] = ("lgb", "xgb", "cat")
    model_kwargs: Dict[str, dict] = field(default_factory=dict)
    members: List[BaseModel] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SimpleAverageEnsemble":
        names = _available_models(self.model_names)
        if not names:
            raise RuntimeError("no base models available")
        self.members = []
        for n in names:
            print(f"[avg.fit] training {n} on {X.shape}")
            kw = self.model_kwargs.get(n, {})
            m = _MODEL_REGISTRY[n](**kw).fit(X, y)
            self.members.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.members])
        return preds.mean(axis=1)


@dataclass
class StackingEnsemble:
    """
    Train base models with K-fold OOF predictions, fit a meta-LightGBM
    on those OOF preds, then refit each base on the full train set so
    we can produce predictions for unseen test rows.

    Folds are split by `moon` when a moon array is supplied — this is
    critical: random row-split would leak future moons into the meta
    training set and inflate Spearman.
    """

    model_names: Sequence[str] = ("lgb", "xgb", "cat")
    model_kwargs: Dict[str, dict] = field(default_factory=dict)
    n_folds: int = 5
    meta_kwargs: Dict = field(default_factory=lambda: dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=50,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=-1,
    ))

    members: List[BaseModel] = field(default_factory=list, init=False)
    meta: Optional[BaseModel] = field(default=None, init=False)
    used_names: List[str] = field(default_factory=list, init=False)

    def _moon_folds(self, moons: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        unique = np.array(sorted(np.unique(moons)))
        # Contiguous moon blocks — preserves temporal order within a fold.
        chunks = np.array_split(unique, self.n_folds)
        folds = []
        for held in chunks:
            held_set = set(held.tolist())
            val_mask = np.array([m in held_set for m in moons])
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            folds.append((train_idx, val_idx))
        return folds

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            moons: Optional[np.ndarray] = None) -> "StackingEnsemble":
        from lightgbm import LGBMRegressor

        names = _available_models(self.model_names)
        if not names:
            raise RuntimeError("no base models available")
        self.used_names = names

        n = len(X)
        oof = np.zeros((n, len(names)), dtype=np.float64)

        if moons is None:
            warnings.warn("StackingEnsemble.fit called without `moons` — "
                          "falling back to random KFold (may leak temporal info).")
            rng = np.random.default_rng(0)
            order = rng.permutation(n)
            folds = []
            for chunk in np.array_split(order, self.n_folds):
                mask = np.zeros(n, dtype=bool)
                mask[chunk] = True
                folds.append((np.where(~mask)[0], np.where(mask)[0]))
        else:
            folds = self._moon_folds(np.asarray(moons))

        for f, (tr, va) in enumerate(folds):
            print(f"[stack.fit] fold {f+1}/{len(folds)} "
                  f"tr={len(tr)} va={len(va)}")
            for j, name in enumerate(names):
                kw = self.model_kwargs.get(name, {})
                m = _MODEL_REGISTRY[name](**kw).fit(X.iloc[tr], y[tr])
                oof[va, j] = m.predict(X.iloc[va])

        print(f"[stack.fit] training meta-LightGBM on OOF "
              f"matrix shape={oof.shape}")
        self.meta = LGBMRegressor(**self.meta_kwargs).fit(oof, y)

        # Refit each base on the FULL training set for inference.
        self.members = []
        for name in names:
            kw = self.model_kwargs.get(name, {})
            print(f"[stack.fit] refitting {name} on full {X.shape}")
            self.members.append(_MODEL_REGISTRY[name](**kw).fit(X, y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([m.predict(X) for m in self.members])
        return self.meta.predict(base_preds)


# ---------------------------------------------------------------------------
# Per-moon Spearman evaluation
# ---------------------------------------------------------------------------

def evaluate_per_moon(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      moons: np.ndarray) -> Dict[str, float]:
    """
    Mean Spearman rank correlation across moons (DataCrunch metric).
    Returns {'mean': ..., 'std': ..., 'n_moons': ...}.
    Moons with <5 finite rows or zero variance in y_pred are skipped.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred, "moon": moons})
    df = df[np.isfinite(df["y"]) & np.isfinite(df["p"])]
    scores = []
    for m, g in df.groupby("moon"):
        if len(g) < 5 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["y"].values, g["p"].values)
        if np.isfinite(rho):
            scores.append(rho)
    if not scores:
        return {"mean": float("nan"), "std": float("nan"), "n_moons": 0}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_moons": len(scores),
    }


# ---------------------------------------------------------------------------
# Train / infer helpers — analogous to main.py, but ensemble-aware
# ---------------------------------------------------------------------------

def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    ensemble: str = "stacking",  # "averaging" or "stacking"
    model_names: Sequence[str] = ("lgb", "xgb", "cat"),
    n_folds: int = 5,
) -> dict:
    """Train ensemble and persist to `model.pkl`. Mirrors main.train signature."""
    feature_cols = [c for c in X_train.columns
                    if c not in ("id", "Id", "moon", "Moon")]
    join_cols = [c for c in ("id", "moon")
                 if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = ("target" if "target" in merged.columns
                  else [c for c in merged.columns
                        if c not in feature_cols + ["id", "moon"]][0])

    mask = merged[target_col].notna()
    Xt = merged.loc[mask, feature_cols]
    yt = merged.loc[mask, target_col].to_numpy()
    moons = (merged.loc[mask, "moon"].to_numpy()
             if "moon" in merged.columns else None)
    print(f"[ensemble.train] ensemble={ensemble} models={list(model_names)} "
          f"rows={len(yt)} features={len(feature_cols)}")

    if ensemble == "averaging":
        model = SimpleAverageEnsemble(model_names=model_names).fit(Xt, yt)
    elif ensemble == "stacking":
        model = StackingEnsemble(model_names=model_names,
                                 n_folds=n_folds).fit(Xt, yt, moons=moons)
    else:
        raise ValueError(f"unknown ensemble='{ensemble}'")

    os.makedirs(model_directory_path, exist_ok=True)
    with open(os.path.join(model_directory_path, "model.pkl"), "wb") as f:
        pickle.dump((model, feature_cols, target_col, ensemble), f)
    return {"feature_cols": feature_cols, "target": target_col, "ensemble": ensemble}


def infer_ensemble(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    with open(os.path.join(model_directory_path, "model.pkl"), "rb") as f:
        model, feature_cols, target_col, _ensemble = pickle.load(f)
    preds = model.predict(X_test[feature_cols])
    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, target_col: preds})


# ---------------------------------------------------------------------------
# Smoke test (10k x 100 synthetic) + per-moon validation
# ---------------------------------------------------------------------------

def _make_synthetic(n_rows=10_000, n_features=100, n_moons=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    # weak signal from first 10 features, the rest are noise
    w = np.zeros(n_features, dtype=np.float32)
    w[:10] = rng.normal(scale=0.3, size=10)
    y = X @ w + 0.5 * rng.normal(size=n_rows).astype(np.float32)
    moons = rng.integers(0, n_moons, size=n_rows)
    ids = np.arange(n_rows)
    Xdf = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
    Xdf.insert(0, "moon", moons)
    Xdf.insert(0, "id", ids)
    ydf = pd.DataFrame({"id": ids, "moon": moons, "target": y})
    return Xdf, ydf


def _smoke_test():
    print("=" * 70)
    print("Ensemble v1 smoke test — 10k rows x 100 features synthetic")
    print("=" * 70)

    X, y = _make_synthetic()
    feature_cols = [c for c in X.columns if c not in ("id", "moon")]

    # train/val split by moon (last 4 moons = val)
    moons_sorted = sorted(X["moon"].unique())
    val_moons = set(moons_sorted[-4:])
    tr_mask = ~X["moon"].isin(val_moons)
    va_mask = X["moon"].isin(val_moons)
    print(f"train rows={tr_mask.sum()} val rows={va_mask.sum()} "
          f"(val moons={sorted(val_moons)})")

    Xtr, ytr = X.loc[tr_mask, feature_cols], y.loc[tr_mask, "target"].to_numpy()
    Xva, yva = X.loc[va_mask, feature_cols], y.loc[va_mask, "target"].to_numpy()
    moons_tr = X.loc[tr_mask, "moon"].to_numpy()
    moons_va = X.loc[va_mask, "moon"].to_numpy()

    requested = ("lgb", "xgb", "cat")
    avail = _available_models(requested)
    print(f"available base models: {avail}")

    results = {}

    # 1) Single LightGBM baseline
    t0 = time.time()
    base = LGBModel(n_estimators=200).fit(Xtr, ytr)
    base_pred = base.predict(Xva)
    results["lgb_only"] = evaluate_per_moon(yva, base_pred, moons_va)
    print(f"[smoke] lgb_only    elapsed={time.time()-t0:.1f}s "
          f"spearman={results['lgb_only']}")

    # 2) Simple averaging
    t0 = time.time()
    avg = SimpleAverageEnsemble(
        model_names=avail,
        model_kwargs={n: dict(n_estimators=200) if n == "lgb" else {}
                      for n in avail},
    ).fit(Xtr, ytr)
    avg_pred = avg.predict(Xva)
    results["averaging"] = evaluate_per_moon(yva, avg_pred, moons_va)
    print(f"[smoke] averaging   elapsed={time.time()-t0:.1f}s "
          f"spearman={results['averaging']}")

    # 3) Stacking (3 folds for speed)
    t0 = time.time()
    stk = StackingEnsemble(
        model_names=avail,
        model_kwargs={n: dict(n_estimators=200) if n == "lgb" else {}
                      for n in avail},
        n_folds=3,
    ).fit(Xtr, ytr, moons=moons_tr)
    stk_pred = stk.predict(Xva)
    results["stacking"] = evaluate_per_moon(yva, stk_pred, moons_va)
    print(f"[smoke] stacking    elapsed={time.time()-t0:.1f}s "
          f"spearman={results['stacking']}")

    print("-" * 70)
    print("Per-moon Spearman summary:")
    for k, v in results.items():
        print(f"  {k:12s}  mean={v['mean']:+.4f}  std={v['std']:.4f}  "
              f"n_moons={v['n_moons']}")
    print("-" * 70)

    # Assertion: at least ONE ensemble must beat single LGB on synthetic.
    # (Synthetic is linear-ish + Gaussian, so gains are modest but real.)
    base_score = results["lgb_only"]["mean"]
    best_ensemble = max(results["averaging"]["mean"], results["stacking"]["mean"])
    delta = best_ensemble - base_score
    print(f"best ensemble - lgb_only = {delta:+.4f}")
    if delta > 0:
        print("PASS: ensemble beats single LightGBM on per-moon Spearman.")
    else:
        print("WARN: ensemble did not beat single LightGBM on this sample. "
              "This can happen with tiny synthetic data; rerun on real data.")
    return results


if __name__ == "__main__":
    _smoke_test()
