"""Comprehensive environment verification on GPU server.

Run after SSH connection established + venv created.
Aborts on first failure with clear error message.
"""
import os
import sys
import subprocess


def step(name):
    print(f"\n{'='*60}\n[STEP] {name}\n{'='*60}")


def fail(msg):
    print(f"❌ FAIL: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"✓ {msg}")


step("1. System info")
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

step("2. Hardware: GPU")
try:
    import torch
    if not torch.cuda.is_available():
        fail("CUDA not available")
    n = torch.cuda.device_count()
    ok(f"CUDA available, {n} device(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        ok(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f} GB")
    ok(f"PyTorch CUDA: {torch.version.cuda}")
except ImportError:
    fail("torch not installed")

step("3. Hardware: CPU + RAM")
import multiprocessing
print(f"CPU count: {multiprocessing.cpu_count()}")
try:
    with open('/proc/meminfo') as f:
        mem = {l.split(':')[0]: l.split(':')[1].strip() for l in f.readlines()[:3]}
    print(f"Memory: {mem}")
except Exception as e:
    print(f"  (mem info skipped: {e})")

step("4. Disk")
print(subprocess.check_output(['df', '-h', '/workspace']).decode())

step("5. ML libs: xgboost")
try:
    import xgboost
    ok(f"xgboost {xgboost.__version__}")
    if int(xgboost.__version__.split('.')[0]) < 2:
        fail(f"xgboost too old, need >= 2.0")
except ImportError:
    fail("xgboost not installed")

step("6. ML libs: lightgbm")
try:
    import lightgbm
    ok(f"lightgbm {lightgbm.__version__}")
except ImportError:
    fail("lightgbm not installed")

step("7. ML libs: catboost")
try:
    import catboost
    ok(f"catboost {catboost.__version__}")
except ImportError:
    fail("catboost not installed (may be optional)")

step("8. Data libs")
import pandas as pd, numpy as np
ok(f"pandas {pd.__version__}, numpy {np.__version__}")

step("9. XGB GPU sanity (10k rows, 100 feats)")
import xgboost as xgb
X = np.random.randn(10000, 100).astype(np.float32)
y = np.random.randn(10000).astype(np.float32)
import time
t0 = time.time()
m = xgb.XGBRegressor(n_estimators=50, tree_method='hist', device='cuda:0', verbosity=0)
m.fit(X, y)
dt = time.time() - t0
ok(f"XGB GPU train OK in {dt:.2f}s")

step("10. PyTorch GPU sanity")
import torch
x = torch.randn(1024, 1024, device='cuda:0')
y = x @ x.T
torch.cuda.synchronize()
ok(f"torch GPU matmul OK, result {y.shape}")

step("11. Data files")
data_dir = '/workspace/data'
if not os.path.isdir(data_dir):
    fail(f"{data_dir} missing")
for f in ['X.reduced.parquet', 'y.reduced.parquet']:
    path = os.path.join(data_dir, f)
    if not os.path.exists(path):
        fail(f"{path} missing")
    sz = os.path.getsize(path) / 1e6
    ok(f"{f}: {sz:.1f} MB")

step("12. Data integrity")
X = pd.read_parquet('/workspace/data/X.reduced.parquet')
y_df = pd.read_parquet('/workspace/data/y.reduced.parquet')
if X.shape != (1637276, 1152):
    fail(f"X shape mismatch: {X.shape}, expected (1637276, 1152)")
if y_df.shape != (1637276, 3):
    fail(f"y shape mismatch: {y_df.shape}, expected (1637276, 3)")
ok(f"X={X.shape}, y={y_df.shape}, X dtypes={X.dtypes.value_counts().to_dict()}")
ok(f"moon range: X=[{X['moon'].min()}, {X['moon'].max()}], y=[{y_df['moon'].min()}, {y_df['moon'].max()}]")

print("\n🎉 ALL CHECKS PASSED. Ready to run experiments.")
