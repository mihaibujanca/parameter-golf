#!/bin/bash
# RunPod setup script — run once after pod starts
# Template: runpod/parameter-golf (https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j)
#
# 1. Start a pod with the template above (1xH100 is fine for data analysis, cheaper)
# 2. SSH into the pod
# 3. Run this script: bash scripts/runpod_setup.sh

set -e

echo "=== RunPod Setup ==="

# Clone our repo
if [ ! -d /workspace/parameter-golf ]; then
    echo "Cloning repo..."
    cd /workspace
    git clone https://github.com/mihaibujanca/parameter-golf.git
    cd parameter-golf
else
    echo "Repo exists, pulling latest..."
    cd /workspace/parameter-golf
    git pull
fi

# Download data if not present
if [ ! -d /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 ]; then
    echo "Downloading FineWeb data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo "Data already present"
    ls -la data/datasets/fineweb10B_sp1024/ | head -5
    echo "..."
    ls data/datasets/fineweb10B_sp1024/*.bin | wc -l
    echo "shards total"
fi

# Install any missing deps
echo "Checking dependencies..."
pip install brotli sentencepiece zstandard 2>/dev/null | tail -1

# Verify setup
echo ""
echo "=== Verification ==="
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import sentencepiece; print('sentencepiece: OK')
import zstandard; print('zstandard: OK')
import brotli; print('brotli: OK')
"

# Count data
echo ""
echo "=== Data ==="
python3 -c "
import glob, os, numpy as np
train = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
val = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
print(f'Train shards: {len(train)}')
print(f'Val shards: {len(val)}')
if train:
    t0 = np.memmap(train[0], dtype=np.uint16, mode='r')
    print(f'First train shard: {len(t0):,} tokens ({len(t0)*2/1e6:.1f} MB)')
    total_train = sum(os.path.getsize(f)//2 for f in train)
    total_val = sum(os.path.getsize(f)//2 for f in val)
    print(f'Total train: {total_train:,} tokens ({total_train/1e9:.2f}B)')
    print(f'Total val: {total_val:,} tokens ({total_val/1e6:.1f}M)')
"

echo ""
echo "=== Ready ==="
echo "cd /workspace/parameter-golf"
echo "Run data analysis: python3 scripts/data_analysis.py"
echo "Run training: bash records/track_10min_16mb/2026-03-30_MixedPrec_PTQCorrection/launch.sh"
