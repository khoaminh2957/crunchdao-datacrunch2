"""R12: MLP with LayerNorm (not BN), small batch, train carefully on k=9 test.

Hypothesis: NN was failing before due to BN+per-moon-zscore conflict.
This uses LayerNorm + raw features (no z-score).
"""
import pandas as pd, numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import pearsonr

X = pd.read_parquet("/workspace/data/X.reduced.parquet")
y = pd.read_parquet("/workspace/data/y.reduced.parquet")
feat = [c for c in X.columns if c not in ('id','moon')]
merged = X.merge(y, on=['id','moon'], how='inner')

TRAIN_ENDS = [600, 650, 700, 750, 770]
import torch, torch.nn as nn
device = torch.device('cuda:0')

def k9(name, fn):
    scores = []
    for T in TRAIN_ENDS:
        tm = T + 9
        if tm > merged['moon'].max(): continue
        tr = merged[merged['moon'] <= T]
        yt = tr['target'].values.astype(np.float32)
        Xt = tr[feat].values.astype(np.float32)
        test = merged[merged['moon'] == tm]
        Xte = test[feat].values.astype(np.float32)
        y_te = test['target'].values
        t0 = time.time()
        pred = fn(Xt, yt, Xte)
        r = pearsonr(pred, y_te)[0]
        scores.append(r)
        print(f"  [{name}] T={T} pearson={r:+.5f} ({time.time()-t0:.0f}s)", flush=True)
    return scores

def train_mlp(Xt, yt, Xte):
    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    torch.manual_seed(42)
    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(yt).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    model = MLP(Xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    BATCH = 4096
    EPOCHS = 10
    n = len(Xt_t)
    model.train()
    for ep in range(EPOCHS):
        idx = torch.randperm(n, device=device)
        total = 0
        for b0 in range(0, n, BATCH):
            sub = idx[b0:b0+BATCH]
            opt.zero_grad()
            pred = model(Xt_t[sub])
            loss = loss_fn(pred, yt_t[sub])
            loss.backward(); opt.step()
            total += loss.item() * len(sub)
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).cpu().numpy()
    return preds


print("=== R12 MLP LayerNorm ===")
results = k9('mlp_ln', train_mlp)
print(f"\nMean: {np.mean(results):+.5f}, std: {np.std(results):.5f}")
print(f"Folds: {[round(s, 4) for s in results]}")
