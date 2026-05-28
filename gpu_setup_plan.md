# GPU Server Setup Plan — 2x RTX 5060 Ti

## Instance specs
- 2x RTX 5060 Ti (31.9 GB VRAM total, 15.95 GB each)
- 192 vCPU AMD EPYC 7K62 48-Core
- 128 GB RAM
- 32 GB disk (small! data has X=700MB + y=7MB = ~700MB)
- CUDA 12.9 (max 13.1)
- Base image: vastai/sd-forge_neo (stable-diffusion image — will have torch + cuda)
- IP: 141.0.85.210, SSH port: PENDING USER

## Pre-flight checks (run in sequence, abort on any failure)
1. SSH handshake works
2. `nvidia-smi` shows 2 GPUs
3. `python --version` >= 3.10
4. `df -h /workspace` >= 5 GB free
5. `free -h` confirms 128 GB RAM
6. `cat /etc/resolv.conf` confirms DNS works

## Required libraries (install in venv)
```
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install pandas==2.2.* numpy==1.26.* scipy
pip install lightgbm xgboost catboost
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pyarrow pyarrow-cython
```

## Data transfer
- Source local: `C:\Users\Admin\Downloads\X.reduced.parquet` (700 MB) + `y.reduced.parquet` (7 MB)
- Method: `scp -P <port> X.reduced.parquet y.reduced.parquet root@141.0.85.210:/workspace/data/`
- Verify size match after transfer

## Verification scripts (run before any model training)
```python
# verify_env.py
import torch, xgboost, lightgbm, catboost, pandas as pd, numpy as np
assert torch.cuda.is_available(), "CUDA missing"
assert torch.cuda.device_count() == 2, "Expected 2 GPUs"
print(f"PyTorch CUDA: {torch.version.cuda}, devices: {torch.cuda.device_count()}")
print(f"XGB: {xgboost.__version__}, LGBM: {lightgbm.__version__}, CB: {catboost.__version__}")

# Test XGB on GPU
import xgboost as xgb
X = np.random.randn(10000, 100).astype(np.float32)
y = np.random.randn(10000).astype(np.float32)
m = xgb.XGBRegressor(n_estimators=10, tree_method='hist', device='cuda:0')
m.fit(X, y)
print("XGB GPU train OK")

# Verify data
X = pd.read_parquet('/workspace/data/X.reduced.parquet')
assert X.shape == (1637276, 1152), f"X shape mismatch: {X.shape}"
y = pd.read_parquet('/workspace/data/y.reduced.parquet')
assert y.shape == (1637276, 3), f"y shape mismatch"
print(f"Data verified: X={X.shape}, y={y.shape}")
```

## What to compute on GPU
1. **Proper walk-forward CV**: train 1-200/test 201-300, 1-300/test 301-400, …, 1-700/test 701-781
   - Cost: ~6 folds × 5 min XGB = 30 min for one config
   - Score = mean across folds, std = stability
2. **Hyperparameter sweep**: 40-60 XGB configs ranking by mean fold Spearman
3. **Pre-screen variants locally** before pushing cloud (avoid wasting quota)

## Risk checklist
- [ ] SSH port provided by user
- [ ] GPU shows 0% util before our work starts (no other tenant)
- [ ] /workspace has >5GB free
- [ ] PyTorch CUDA version matches CUDA driver (12.9 driver supports cu121 wheel)
- [ ] xgboost.__version__ >= 2.0 (else `device='cuda'` arg unknown)
- [ ] catboost works with the cuda version
- [ ] No background processes (sd-forge) competing for GPU memory

## Pending from user
- SSH port number (likely 5-digit like 12345)
- Optionally: shell type bash/zsh, prefer venv vs conda
