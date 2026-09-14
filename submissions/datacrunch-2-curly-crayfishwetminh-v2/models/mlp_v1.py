"""CrunchDAO DataCrunch #2 — MLP baseline (orthogonal to tree models).

Architecture: [1150 -> 256 -> 128 -> 1] with BatchNorm + ReLU + Dropout(0.3).
Training: median-impute NaN -> standardize (running mean/std) -> MSE on per-moon
rank target in [0, 1] -> Adam(lr=1e-3), batch 4096, 10 epochs (CPU-friendly).

Persists a single model.pkl with state_dict + imputer stats + scaler stats +
feature_cols + arch via pickle.

requires torch>=2.0
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Model -----------------------------

class MLPv1(nn.Module):
    """1150 -> 256 -> 128 -> 1 with BN + ReLU + Dropout(0.3)."""

    def __init__(self, in_dim: int, hidden1: int = 256, hidden2: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------- Helpers -----------------------------

def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("id", "Id", "moon", "Moon", "target", "Target")]


def _fit_median_impute(X: np.ndarray) -> np.ndarray:
    """Per-feature median, NaN-safe. Fallback to 0 for all-NaN columns."""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float32)


def _apply_median_impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Replace NaN / +-inf with column median."""
    Xi = np.where(np.isfinite(X), X, med).astype(np.float32, copy=False)
    return Xi


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running mean/std on the already-imputed matrix."""
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sd
    return Xs.astype(np.float32, copy=False)


def _rank_per_moon(y_df: pd.DataFrame, moon_col: str, target_col: str) -> pd.Series:
    """Per-moon percentile rank in [0, 1] (Spearman-native target)."""
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


# ----------------------------- Train / Infer -----------------------------

def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    *,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 0,
):
    """Train MLPv1 on per-moon rank target with MSE.

    Persists model.pkl with state_dict, imputer median, scaler mu/sd,
    feature_cols, target_col, and arch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feature_cols = _feature_cols(X_train)
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")

    target_col = "target" if "target" in merged.columns else [
        c for c in merged.columns if c not in feature_cols + ["id", "moon", "Id", "Moon"]
    ][0]
    moon_col = "moon" if "moon" in merged.columns else "Moon"

    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)

    # Per-moon rank target in [0, 1]
    y_ranked = _rank_per_moon(
        merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col
    ).to_numpy(dtype=np.float32, copy=False)

    Xt = merged[feature_cols].to_numpy(dtype=np.float32, copy=False)
    print(f"[mlp_v1.train] device={device_t} rows={len(Xt)} features={len(feature_cols)}")

    # 1) Median impute (fit + apply)
    med = _fit_median_impute(Xt)
    Xi = _apply_median_impute(Xt, med)

    # 2) Standardize (fit + apply) on imputed data
    mu, sd = _fit_standardizer(Xi)
    Xs = _apply_standardizer(Xi, mu, sd)

    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y_ranked))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = MLPv1(in_dim=len(feature_cols)).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device_t, non_blocking=True)
            yb = yb.to(device_t, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        print(f"[mlp_v1.train] epoch {ep:02d}/{epochs} train_mse={tot_loss / max(n_seen, 1):.6f}")

    os.makedirs(model_directory_path, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_cols": feature_cols,
        "target_col": target_col,
        "imputer": {"median": med},
        "scaler": {"mu": mu, "sd": sd},
        "arch": {"in_dim": len(feature_cols), "hidden1": 256, "hidden2": 128, "p_drop": 0.3},
    }
    with open(os.path.join(model_directory_path, "model.pkl"), "wb") as f:
        pickle.dump(payload, f)
    print(f"[mlp_v1.train] saved model.pkl ({len(feature_cols)} features)")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    *,
    batch_size: int = 16384,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Predict the per-moon rank target.

    Returns DataFrame with columns [id, moon, prediction] (non-negotiable).
    """
    with open(os.path.join(model_directory_path, "model.pkl"), "rb") as f:
        payload = pickle.load(f)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_cols = payload["feature_cols"]
    med = payload["imputer"]["median"]
    mu = payload["scaler"]["mu"]
    sd = payload["scaler"]["sd"]
    arch = payload["arch"]

    model = MLPv1(**arch).to(device_t)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    X = X_test[feature_cols].to_numpy(dtype=np.float32, copy=False)
    Xi = _apply_median_impute(X, med)
    Xs = _apply_standardizer(Xi, mu, sd)

    preds = np.empty(len(Xs), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            xb = torch.from_numpy(Xs[i : i + batch_size]).to(device_t)
            preds[i : i + batch_size] = model(xb).cpu().numpy()

    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    out = pd.DataFrame({
        "id": X_test[id_col].values,
        "moon": X_test[moon_col].values,
        "prediction": preds,
    })
    return out


# ----------------------------- Smoke test -----------------------------

def _smoke_test(n_rows: int = 1000, n_features: int = 1150, n_moons: int = 5, seed: int = 0) -> None:
    """Schema-only smoke: build synthetic data, train tiny model, verify infer schema."""
    import tempfile

    rng = np.random.default_rng(seed)
    moons = rng.integers(0, n_moons, size=n_rows).astype(np.int32)
    ids = np.arange(n_rows, dtype=np.int64)
    feats = rng.standard_normal(size=(n_rows, n_features)).astype(np.float32)
    # Inject some NaNs to exercise the imputer
    nan_mask = rng.random(size=feats.shape) < 0.01
    feats[nan_mask] = np.nan

    feat_cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(feats, columns=feat_cols)
    X.insert(0, "moon", moons)
    X.insert(0, "id", ids)

    # Synthetic continuous target — train() will per-moon-rank it
    y = pd.DataFrame({
        "id": ids,
        "moon": moons,
        "target": rng.standard_normal(n_rows).astype(np.float32),
    })

    with tempfile.TemporaryDirectory() as tmp:
        train(X, y, tmp, epochs=2, batch_size=256)
        preds = infer(X, tmp)
        assert list(preds.columns) == ["id", "moon", "prediction"], \
            f"infer schema must be [id, moon, prediction], got {list(preds.columns)}"
        assert len(preds) == n_rows, f"row count mismatch: {len(preds)} vs {n_rows}"
        assert preds["prediction"].notna().all(), "predictions contain NaN"
        print(f"[mlp_v1.smoke] OK rows={len(preds)} cols={list(preds.columns)}")


if __name__ == "__main__":
    _smoke_test()
