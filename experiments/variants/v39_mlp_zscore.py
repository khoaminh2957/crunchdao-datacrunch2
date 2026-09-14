"""v39: PyTorch MLP on raw + per-moon z-score features (2300 inputs).

v30 XGB(raw+z) = 0.0563.
v39 tests if MLP can match/beat — different inductive bias (smooth non-linear vs trees).

Architecture: [2300 → 512 → 256 → 128 → 1] with BatchNorm + ReLU + Dropout(0.3).
CPU-friendly: batch=4096, epochs=15, Adam lr=1e-3.

requires torch>=2.0
"""
import pandas as pd
import numpy as np
import pickle
import os


def _rank_per_moon_series(y_df, moon_col, target_col):
    grp = y_df.groupby(moon_col)[target_col]
    return (grp.rank(method="average", na_option="keep") / grp.transform("count")).astype(np.float32)


def _zscore_per_moon(X_df, feature_cols, moon_col, eps=1e-9):
    out = X_df[[moon_col] + feature_cols].copy()
    mean = out.groupby(moon_col)[feature_cols].transform("mean")
    std = out.groupby(moon_col)[feature_cols].transform("std").replace(0, np.nan)
    return ((out[feature_cols] - mean) / (std + eps)).fillna(0.0).astype(np.float32)


def _build_features(X_df, raw_feats, moon_col):
    Xz = _zscore_per_moon(X_df, raw_feats, moon_col=moon_col)
    Xz.columns = [f"{c}_z" for c in raw_feats]
    return pd.concat([X_df[raw_feats].reset_index(drop=True),
                      Xz.reset_index(drop=True)], axis=1)


def train(X_train, y_train, model_directory_path):
    import torch
    import torch.nn as nn
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    raw_feats = [c for c in X_train.columns if c not in ("id", "Id", "moon", "Moon")]
    moon_col = "moon" if "moon" in X_train.columns else "Moon"
    id_col = "id" if "id" in X_train.columns else "Id"

    Xc = _build_features(X_train, raw_feats, moon_col=moon_col)
    Xc[id_col] = X_train[id_col].values
    Xc[moon_col] = X_train[moon_col].values
    feature_cols = raw_feats + [f"{c}_z" for c in raw_feats]
    n_feats = len(feature_cols)

    merged = Xc.merge(y_train, on=[c for c in ("id","moon") if c in Xc.columns and c in y_train.columns], how="inner")
    target_col = "target" if "target" in merged.columns else [c for c in y_train.columns if c not in ("id","moon")][0]
    mask = merged[target_col].notna()
    merged = merged.loc[mask].reset_index(drop=True)
    y_ranked = _rank_per_moon_series(merged[[moon_col, target_col]], moon_col=moon_col, target_col=target_col).values

    # Fit imputer (median) + scaler (mean/std) on training
    X_arr = merged[feature_cols].values.astype(np.float32)
    median = np.nanmedian(X_arr, axis=0)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(median, inds[1])
    mu = X_arr.mean(axis=0)
    sd = X_arr.std(axis=0) + 1e-6
    X_arr = (X_arr - mu) / sd

    print(f"[v39/train] MLP on {len(X_arr)} rows x {n_feats} feats; arch [2300->512->256->128->1]")

    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = MLP(n_feats)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    BATCH = 4096
    EPOCHS = 15
    Xt = torch.from_numpy(X_arr)
    yt = torch.from_numpy(y_ranked.astype(np.float32))
    n = len(Xt)
    rng = np.random.RandomState(42)

    model.train()
    for ep in range(EPOCHS):
        idx = rng.permutation(n)
        total_loss = 0.0
        for b0 in range(0, n, BATCH):
            sub = idx[b0:b0+BATCH]
            xb = Xt[sub]; yb = yt[sub]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(sub)
        print(f"[v39/train] epoch {ep+1}/{EPOCHS} loss={total_loss/n:.5f}")

    model.eval()
    os.makedirs(model_directory_path, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "median": median,
        "mu": mu, "sd": sd,
        "feature_cols": feature_cols,
        "raw_feats": raw_feats,
        "target_col": target_col,
        "moon_col": moon_col,
        "n_feats": n_feats,
    }
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump(state, f)
    print(f"[v39/train] saved MLP state + feature cols")


def infer(X_test, model_directory_path):
    import torch
    import torch.nn as nn
    torch.set_num_threads(min(16, os.cpu_count() or 8))

    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        s = pickle.load(f)

    Xc = _build_features(X_test, s["raw_feats"], moon_col=s["moon_col"])
    X_arr = Xc[s["feature_cols"]].values.astype(np.float32)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(s["median"], inds[1])
    X_arr = (X_arr - s["mu"]) / s["sd"]

    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = MLP(s["n_feats"])
    model.load_state_dict(s["state_dict"])
    model.eval()

    Xt = torch.from_numpy(X_arr)
    with torch.no_grad():
        # Predict in batches to avoid OOM
        preds = []
        BATCH = 8192
        for b0 in range(0, len(Xt), BATCH):
            preds.append(model(Xt[b0:b0+BATCH]).cpu().numpy())
        preds = np.concatenate(preds)

    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = s["moon_col"]
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
