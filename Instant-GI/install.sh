#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

conda create -n instant-gi python=3.12 -y 
conda activate instant-gi
#export CUDA_VISIBLE_DEVICES='0'

pip install uv

uv pip install --system torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install --system cupy-cuda12x --pre -U -f https://pip.cupy.dev/pre --no-build-isolation

uv pip install --system --no-build-isolation --force-reinstall --no-cache-dir \
    ./submodules/ellipse_fit \
    ./submodules/torch_dither \
    ./submodules/gsplat

uv pip install --system -r requirements.txt --no-build-isolation
mkdir -p checkpoints