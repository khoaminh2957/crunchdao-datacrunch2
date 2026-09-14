"""CrunchDAO DataCrunch #2 — NN alternative to LightGBM baseline.

3-layer MLP (1150 -> 512 -> 128 -> 1) with BatchNorm + GELU + Dropout(0.3).
Adam, lr=1e-3, 20 epochs, batch 8192.

This file mirrors the train/infer interface of main.py so it can be swapped
in for evaluation. It also exposes a __main__ block that:
  * loads the local reduced parquet,
  * trains on a 100k-row sample (train moons only),
  * validates on holdout moons with per-moon Spearman correlation,
  * prints mean per-moon Spearman vs. a quick LightGBM baseline.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Model -----------------------------

class TabularMLP(nn.Module):
    """1150 -> 512 -> 128 -> 1 with BN + GELU + Dropout(0.3)."""

    def __init__(self, in_dim: int, hidden1: int = 512, hidden2: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ----------------------------- Helpers -----------------------------

def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("id", "Id", "moon", "Moon")]


def _fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0).astype(np.float32)
    sd = np.nanstd(X, axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sd
    # NaN-safe — replace with 0 after standardization (= mean imputation)
    np.nan_to_num(Xs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs.astype(np.float32)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r on ranks."""
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def per_moon_spearman(df_preds: pd.DataFrame, pred_col: str, target_col: str, moon_col: str = "moon") -> Tuple[float, pd.Series]:
    """Return (mean_spearman, per_moon_spearman_series)."""
    rows = []
    for m, g in df_preds.groupby(moon_col):
        rho = _spearman(g[pred_col].values, g[target_col].values)
        rows.append((int(m), rho))
    s = pd.Series({m: r for m, r in rows}).sort_index()
    return float(np.nanmean(s.values)), s


# ----------------------------- Train / Infer -----------------------------

def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    *,
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
):
    """Train MLP. Persists model.pkl with (state_dict, feature_cols, target_col, mu, sd, arch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feature_cols = _feature_cols(X_train)
    join_cols = [c for c in ("id", "moon") if c in X_train.columns and c in y_train.columns]
    merged = X_train.merge(y_train, on=join_cols, how="inner")
    target_col = "target" if "target" in merged.columns else [
        c for c in merged.columns if c not in feature_cols + ["id", "moon"]
    ][0]

    mask = merged[target_col].notna()
    Xt = merged.loc[mask, feature_cols].to_numpy(dtype=np.float32, copy=False)
    yt = merged.loc[mask, target_col].to_numpy(dtype=np.float32, copy=False)
    print(f"[nn_v1.train] device={device_t} rows={len(yt)} features={len(feature_cols)}")

    mu, sd = _fit_standardizer(Xt)
    Xs = _apply_standardizer(Xt, mu, sd)

    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(yt))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularMLP(in_dim=len(feature_cols)).to(device_t)
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
        print(f"[nn_v1.train] epoch {ep:02d}/{epochs} train_mse={tot_loss / max(n_seen, 1):.6f}")

    os.makedirs(model_directory_path, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_cols": feature_cols,
        "target_col": target_col,
        "mu": mu,
        "sd": sd,
        "arch": {"in_dim": len(feature_cols), "hidden1": 512, "hidden2": 128, "p_drop": 0.3},
    }
    with open(os.path.join(model_directory_path, "model.pkl"), "wb") as f:
        pickle.dump(payload, f)
    print(f"[nn_v1.train] saved model.pkl ({len(feature_cols)} features)")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    *,
    batch_size: int = 16384,
    device: str | None = None,
) -> pd.DataFrame:
    """Predict target. Returns DataFrame[id, target]."""
    with open(os.path.join(model_directory_path, "model.pkl"), "rb") as f:
        payload = pickle.load(f)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_cols = payload["feature_cols"]
    mu, sd = payload["mu"], payload["sd"]
    arch = payload["arch"]
    target_col = payload["target_col"]

    model = TabularMLP(**arch).to(device_t)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    X = X_test[feature_cols].to_numpy(dtype=np.float32, copy=False)
    Xs = _apply_standardizer(X, mu, sd)

    preds = np.empty(len(Xs), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            xb = torch.from_numpy(Xs[i : i + batch_size]).to(device_t)
            preds[i : i + batch_size] = model(xb).cpu().numpy()

    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, target_col: preds})


# ----------------------------- Smoke test -----------------------------

def _run_smoke(repo_root: str, sample_n: int = 100_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(repo_root, "data")
    X = pd.read_parquet(os.path.join(data_dir, "X.reduced.parquet"))
    y = pd.read_parquet(os.path.join(data_dir, "y.reduced.parquet"))
    with open(os.path.join(data_dir, "moons_split.json")) as f:
        split = json.load(f)
    train_moons = set(split["train"])
    val_moons = sorted(set(split.get("reduced_local", [])) | set(split.get("reduced_cloud", [])))
    print(f"[smoke] train_moons={len(train_moons)} val_moons={len(val_moons)} -> {val_moons}")

    Xtr_full = X[X["moon"].isin(train_moons)]
    print(f"[smoke] train pool rows={len(Xtr_full)} sampling {sample_n}")
    idx = rng.choice(len(Xtr_full), size=min(sample_n, len(Xtr_full)), replace=False)
    Xtr = Xtr_full.iloc[idx].reset_index(drop=True)
    ytr = y[y["id"].isin(Xtr["id"]) & y["moon"].isin(train_moons)]

    Xva = X[X["moon"].isin(val_moons)].reset_index(drop=True)
    yva = y[y["moon"].isin(val_moons)].reset_index(drop=True)
    print(f"[smoke] Xtr={Xtr.shape} ytr={ytr.shape} Xva={Xva.shape} yva={yva.shape}")

    model_dir = os.path.join(repo_root, "resources", "_nn_v1_smoke")
    train(Xtr, ytr, model_dir, epochs=20, batch_size=8192, lr=1e-3, seed=seed)

    preds = infer(Xva, model_dir)
    target_col = "target" if "target" in yva.columns else preds.columns[-1]
    merged = preds.merge(yva[["id", "moon", target_col]], on="id", suffixes=("_pred", ""))
    merged = merged.rename(columns={f"{target_col}_pred": "pred"})
    merged = merged.dropna(subset=[target_col])
    mean_rho, per_moon = per_moon_spearman(merged, "pred", target_col)
    print("[smoke] per-moon Spearman:")
    for m, r in per_moon.items():
        print(f"  moon {m}: {r:+.5f}")
    print(f"[smoke] MEAN per-moon Spearman (NN) = {mean_rho:+.5f}")

    # Quick LightGBM reference on identical sample/split for sanity
    try:
        from lightgbm import LGBMRegressor
        Xtr_m = Xtr.merge(ytr, on=["id", "moon"], how="inner")
        m = Xtr_m[target_col].notna()
        feats = _feature_cols(Xtr)
        lgb = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_data_in_leaf=100, colsample_bytree=0.5, reg_lambda=2.0,
            verbose=-1, n_jobs=-1,
        )
        lgb.fit(Xtr_m.loc[m, feats], Xtr_m.loc[m, target_col])
        lgb_preds = lgb.predict(Xva[feats])
        lgb_df = pd.DataFrame({"id": Xva["id"].values, "moon": Xva["moon"].values, "pred": lgb_preds})
        lgb_m = lgb_df.merge(yva[["id", target_col]], on="id").dropna(subset=[target_col])
        lgb_mean, lgb_per = per_moon_spearman(lgb_m, "pred", target_col)
        print(f"[smoke] MEAN per-moon Spearman (LGB ref on same 100k) = {lgb_mean:+.5f}")
    except Exception as e:
        print(f"[smoke] LGB reference skipped: {e}")


if __name__ == "__main__":
    _run_smoke(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
