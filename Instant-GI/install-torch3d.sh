set -euo pipefail

eval "$(conda shell.bash hook)"

conda create -n instant-gi-stable python=3.11 -y 
conda activate instant-gi-stable

export PYTHON=$(which python)

# Install Build Tools
python -m pip install --no-cache-dir uv ninja

uv pip install --python "$PYTHON" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install --python "$PYTHON" fvcore iopath

# 5. Compile PyTorch3D from Source (Stable Branch)
# Since we are on Torch 2.5, we build from source to guarantee ABI compatibility.
echo "Building PyTorch3D..."
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
uv pip install --python "$PYTHON" --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@main"

# 6. Install CuPy (Matching CUDA 12.x)
uv pip install --python "$PYTHON" cupy-cuda12x

# 7. Clean & Reinstall Custom Submodules
uv pip install --python "$PYTHON" --no-build-isolation --force-reinstall --no-cache-dir \
    ./submodules/ellipse_fit \
    ./submodules/torch_dither \
    ./submodules/gsplat

# 8. Remaining requirements
uv pip install --python "$PYTHON" -r requirements.txt --no-build-isolation

mkdir -p checkpoints
echo "Installation Complete!"