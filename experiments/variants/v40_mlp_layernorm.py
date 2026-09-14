"""v40: MLP với LayerNorm thay BatchNorm (fix v39 = -0.0478).

v39 used BatchNorm with raw+z features → likely BN running stats conflict
with per-moon z-score during inference. LayerNorm normalizes within sample
so no train-test running-stats drift.

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


def _make_model(n_in):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(128, 1),
    )


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

    X_arr = merged[feature_cols].values.astype(np.float32)
    median = np.nanmedian(X_arr, axis=0)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(median, inds[1])
    mu = X_arr.mean(axis=0)
    sd = X_arr.std(axis=0) + 1e-6
    X_arr = (X_arr - mu) / sd
    print(f"[v40/train] MLP+LayerNorm on {len(X_arr)} rows x {n_feats} feats")

    model = _make_model(n_feats)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    BATCH, EPOCHS = 4096, 10
    Xt = torch.from_numpy(X_arr); yt = torch.from_numpy(y_ranked.astype(np.float32))
    n = len(Xt); rng = np.random.RandomState(42)
    model.train()
    for ep in range(EPOCHS):
        idx = rng.permutation(n)
        tot = 0.0
        for b0 in range(0, n, BATCH):
            sub = idx[b0:b0+BATCH]
            xb = Xt[sub]; yb = yt[sub]
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()
            tot += loss.item() * len(sub)
        print(f"[v40/train] epoch {ep+1}/{EPOCHS} loss={tot/n:.5f}")

    model.eval()
    os.makedirs(model_directory_path, exist_ok=True)
    state = {"state_dict": model.state_dict(), "median": median, "mu": mu, "sd": sd,
             "feature_cols": feature_cols, "raw_feats": raw_feats, "moon_col": moon_col, "n_feats": n_feats}
    with open(f"{model_directory_path}/model.pkl", "wb") as f:
        pickle.dump(state, f)


def infer(X_test, model_directory_path):
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        s = pickle.load(f)
    Xc = _build_features(X_test, s["raw_feats"], moon_col=s["moon_col"])
    X_arr = Xc[s["feature_cols"]].values.astype(np.float32)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(s["median"], inds[1])
    X_arr = (X_arr - s["mu"]) / s["sd"]
    model = _make_model(s["n_feats"]); model.load_state_dict(s["state_dict"]); model.eval()
    Xt = torch.from_numpy(X_arr)
    with torch.no_grad():
        preds = []
        for b0 in range(0, len(Xt), 8192):
            preds.append(model(Xt[b0:b0+8192]).squeeze(-1).cpu().numpy())
        preds = np.concatenate(preds)
    id_col = "id" if "id" in X_test.columns else "Id"
    return pd.DataFrame({id_col: X_test[id_col].values, s["moon_col"]: X_test[s["moon_col"]].values, "prediction": preds})
