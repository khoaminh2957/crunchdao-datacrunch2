#!/bin/bash
# Run on remote GPU server. Fully self-contained setup.
set -e  # abort on any error

echo "==================================================="
echo "STEP 1/9: System info"
echo "==================================================="
uname -a
echo ""
nvidia-smi -L
echo ""
nvidia-smi --query-gpu=name,memory.free,memory.total,utilization.gpu --format=csv
echo ""

echo "==================================================="
echo "STEP 2/9: Disk space"
echo "==================================================="
df -h / /workspace 2>/dev/null || df -h /
FREE_MB=$(df --output=avail / | tail -1)
if [ "$FREE_MB" -lt 5000000 ]; then
    echo "❌ Need >5GB free on /"
    exit 1
fi
echo "✓ Disk OK"

echo "==================================================="
echo "STEP 3/9: Python + pip"
echo "==================================================="
which python3
python3 --version
which pip || apt-get install -y python3-pip
pip --version

echo "==================================================="
echo "STEP 4/9: Create workspace + venv"
echo "==================================================="
mkdir -p /workspace/data
mkdir -p /workspace/code
mkdir -p /workspace/results
cd /workspace
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip --quiet

echo "==================================================="
echo "STEP 5/9: Install ML libs"
echo "==================================================="
pip install --quiet pandas==2.2.3 numpy==1.26.4 scipy pyarrow
pip install --quiet xgboost==2.1.* lightgbm==4.5.* catboost==1.2.*
# PyTorch with CUDA 12.1 (works with driver 12.9)
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121 || pip install --quiet torch
echo "✓ Libs installed"

echo "==================================================="
echo "STEP 6/9: Verify all imports"
echo "==================================================="
python -c "
import pandas, numpy, scipy, xgboost, lightgbm, catboost, torch
import pyarrow
print(f'pandas {pandas.__version__}')
print(f'numpy {numpy.__version__}')
print(f'xgboost {xgboost.__version__}')
print(f'lightgbm {lightgbm.__version__}')
print(f'catboost {catboost.__version__}')
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}, devices {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 2, f'Expected 2 GPUs, got {torch.cuda.device_count()}'
"

echo "==================================================="
echo "STEP 7/9: XGB GPU sanity"
echo "==================================================="
python -c "
import xgboost as xgb, numpy as np, time
X = np.random.randn(50000, 200).astype(np.float32)
y = np.random.randn(50000).astype(np.float32)
t0 = time.time()
m = xgb.XGBRegressor(n_estimators=100, tree_method='hist', device='cuda:0', verbosity=0)
m.fit(X, y)
print(f'✓ XGB GPU 50k×200 / 100 trees: {time.time()-t0:.2f}s')
"

echo "==================================================="
echo "STEP 8/9: Check data files"
echo "==================================================="
if [ ! -f /workspace/data/X.reduced.parquet ]; then
    echo "❌ /workspace/data/X.reduced.parquet missing — run scp transfer first"
    exit 1
fi
if [ ! -f /workspace/data/y.reduced.parquet ]; then
    echo "❌ /workspace/data/y.reduced.parquet missing"
    exit 1
fi
ls -la /workspace/data/

echo "==================================================="
echo "STEP 9/9: Data integrity"
echo "==================================================="
python -c "
import pandas as pd
X = pd.read_parquet('/workspace/data/X.reduced.parquet')
y = pd.read_parquet('/workspace/data/y.reduced.parquet')
assert X.shape == (1637276, 1152), f'X.shape={X.shape}'
assert y.shape == (1637276, 3), f'y.shape={y.shape}'
print(f'✓ X={X.shape}, y={y.shape}, moons in X: [{X[\"moon\"].min()}, {X[\"moon\"].max()}]')
"

echo ""
echo "🎉 ALL 9 SETUP STEPS PASSED. Ready for training."
echo "Next: python /workspace/code/gpu_sweep.py"
