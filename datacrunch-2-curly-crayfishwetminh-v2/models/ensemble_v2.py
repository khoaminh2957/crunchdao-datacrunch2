"""CrunchDAO DataCrunch #2 - Ensemble v2.

Trains multiple sibling models, then averages per-moon RANKED predictions
at inference time. Ranking each member's predictions before averaging makes
the ensemble invariant to scale/offset differences between members
(LGBM regression scores vs ExtraTrees vs CatBoost can live on totally
different magnitudes).

Public API:
    train(X_train, y_train, model_directory_path)
    infer(X_test, model_directory_path) -> pd.DataFrame[id, moon, prediction]

Each member writes its artifacts under model_directory_path/<subdir>/ so
the per-member pickles never collide. Members whose train() raises are
logged and skipped; infer() ignores members whose subdir doesn't have a
model.pkl (i.e. the ones that failed to train).
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
# (subdir_name, module_import_path, source_file_relative_to_repo_root)
# source_file is used to check existence before importing — keeps the
# ensemble runnable even if a member file hasn't been written yet.
_CANDIDATES: List[Tuple[str, str, str]] = [
    ("lgbm",       "models.lgbm_v1",       "models/lgbm_v1.py"),
    ("xgb",        "models.xgb_v1",        "models/xgb_v1.py"),
    ("catboost",   "models.catboost_v1",   "models/catboost_v1.py"),
    ("ridge",      "models.ridge_v1",      "models/ridge_v1.py"),
    ("extratrees", "models.extratrees_v1", "models/extratrees_v1.py"),
    ("dart",       "models.lgbm_dart_v1",  "models/lgbm_dart_v1.py"),
]


def _repo_root() -> Path:
    """models/ensemble_v2.py -> repo root."""
    return Path(__file__).resolve().parent.parent


def _build_members() -> List[Tuple[str, str]]:
    """Filter the candidate list to members whose source file actually exists."""
    root = _repo_root()
    members: List[Tuple[str, str]] = []
    for subdir, mod_path, rel_file in _CANDIDATES:
        if (root / rel_file).is_file():
            members.append((subdir, mod_path))
        else:
            print(f"[ensemble] skip '{subdir}': {rel_file} not found")
    return members


MEMBERS: List[Tuple[str, str]] = _build_members()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ensure_repo_on_path() -> None:
    """Both `main` (top-level) and `models.*` imports require repo root on sys.path."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_member(mod_path: str):
    _ensure_repo_on_path()
    return importlib.import_module(mod_path)


def _rank_per_moon(df: pd.DataFrame) -> pd.Series:
    """Per-moon percentile rank of `prediction`, in (0, 1]. Pure ranks --
    no centering -- so a simple mean across members stays in (0, 1]."""
    grp = df.groupby("moon")["prediction"]
    return (grp.rank(method="average") / grp.transform("count")).astype(np.float64)


# ---------------------------------------------------------------------------
# train / infer
# ---------------------------------------------------------------------------
def train(X_train: pd.DataFrame, y_train, model_directory_path: str) -> None:
    """Train every available member into its own subdirectory. Failures are
    isolated -- one broken member doesn't kill the rest of the ensemble."""
    os.makedirs(model_directory_path, exist_ok=True)

    if not MEMBERS:
        raise RuntimeError("[ensemble] no members available -- check models/ directory")

    print(f"[ensemble] training {len(MEMBERS)} member(s): "
          f"{[name for name, _ in MEMBERS]}")

    trained: List[str] = []
    failed: List[str] = []

    for subdir, mod_path in MEMBERS:
        member_dir = os.path.join(model_directory_path, subdir)
        os.makedirs(member_dir, exist_ok=True)
        print(f"\n[ensemble] === training '{subdir}' ({mod_path}) -> {member_dir} ===")
        try:
            mod = _import_member(mod_path)
            mod.train(X_train, y_train, member_dir)
            # Member's contract says it writes model.pkl -- verify so a silent
            # skip in train() doesn't poison infer() later.
            if not os.path.exists(os.path.join(member_dir, "model.pkl")):
                raise FileNotFoundError(f"{subdir}/model.pkl not written by train()")
            trained.append(subdir)
            print(f"[ensemble] '{subdir}' OK")
        except Exception as e:
            failed.append(subdir)
            print(f"[ensemble] '{subdir}' FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n[ensemble] done. trained={trained} failed={failed}")
    if not trained:
        raise RuntimeError("[ensemble] every member failed -- nothing to infer with")


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    """Score X_test with every member that has a model.pkl on disk, rank each
    member's predictions per moon, then average the ranks. Output schema is
    [id, moon, prediction] -- exactly what the CrunchDAO scorer expects."""
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"

    # Stable per-row key (some members may reorder; ranks must align on (id,moon))
    base = pd.DataFrame({
        "id": X_test[id_col].to_numpy(),
        "moon": X_test[moon_col].to_numpy(),
    })

    rank_frames: List[pd.Series] = []
    used: List[str] = []
    skipped: List[str] = []

    for subdir, mod_path in MEMBERS:
        member_dir = os.path.join(model_directory_path, subdir)
        if not os.path.exists(os.path.join(member_dir, "model.pkl")):
            skipped.append(subdir)
            print(f"[ensemble] '{subdir}': no model.pkl -- skipping")
            continue
        try:
            mod = _import_member(mod_path)
            preds = mod.infer(X_test, member_dir)
            if not {"id", "moon", "prediction"}.issubset(preds.columns):
                raise ValueError(f"{subdir}.infer returned {list(preds.columns)}")
            # Align to base via (id, moon) -- guards against any reordering
            merged = base.merge(
                preds[["id", "moon", "prediction"]],
                on=["id", "moon"], how="left",
            ).copy()
            merged.loc[:, "prediction"] = merged["prediction"].astype(np.float64)
            # Median-fill any NaN before ranking so they don't break the avg
            if merged["prediction"].isna().any():
                merged.loc[:, "prediction"] = merged["prediction"].fillna(
                    merged["prediction"].median()
                )
            ranks = _rank_per_moon(merged)
            ranks.name = subdir
            rank_frames.append(ranks)
            used.append(subdir)
            print(f"[ensemble] '{subdir}': {len(preds)} preds ranked per moon")
        except Exception as e:
            skipped.append(subdir)
            print(f"[ensemble] '{subdir}' INFER FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    if not rank_frames:
        raise RuntimeError("[ensemble] no usable members at infer time")

    # Mean of per-moon percentile ranks -- stays in (0, 1].
    rank_matrix = pd.concat(rank_frames, axis=1)
    mean_rank = rank_matrix.mean(axis=1).astype(np.float64)

    out = pd.DataFrame({
        "id": base["id"].to_numpy(),
        "moon": base["moon"].to_numpy(),
        "prediction": mean_rank.to_numpy(),
    })[["id", "moon", "prediction"]]

    print(f"[ensemble] ensembled {len(used)} member(s): {used}; skipped={skipped}")
    print(f"[ensemble] output shape={out.shape}, "
          f"pred range=[{out['prediction'].min():.4f}, {out['prediction'].max():.4f}]")
    return out


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    n_rows = 1000
    n_feats = 50  # keep small so the smoke test stays quick
    n_moons = 10

    moons = rng.integers(0, n_moons, size=n_rows)
    feats = rng.standard_normal(size=(n_rows, n_feats)).astype(np.float32)

    # Sparse bounded target mimicking real distribution
    target = np.zeros(n_rows, dtype=np.float32)
    nz = rng.random(n_rows) > 0.88
    target[nz] = rng.uniform(-1.0, 1.0, size=nz.sum()).astype(np.float32)

    X = pd.DataFrame(feats, columns=[f"Feature_{i}" for i in range(n_feats)])
    X.insert(0, "moon", moons)
    X.insert(0, "id", np.arange(n_rows))
    y = pd.DataFrame({
        "id": X["id"].to_numpy(),
        "moon": X["moon"].to_numpy(),
        "target": target,
    })

    print(f"[smoke] MEMBERS available: {MEMBERS}")

    with tempfile.TemporaryDirectory() as tmp:
        train(X, y, tmp)
        out = infer(X, tmp)

        assert list(out.columns) == ["id", "moon", "prediction"], \
            f"Bad schema: {list(out.columns)}"
        assert len(out) == n_rows, f"Row count {len(out)} != {n_rows}"
        assert out["prediction"].notna().all(), "NaN in ensemble predictions"
        assert (out["prediction"] >= 0).all() and (out["prediction"] <= 1).all(), \
            "Mean-of-percentile-ranks must be in [0, 1]"
        print(f"OK -- rows={len(out)} "
              f"pred_range=[{out['prediction'].min():.4f}, "
              f"{out['prediction'].max():.4f}]")
