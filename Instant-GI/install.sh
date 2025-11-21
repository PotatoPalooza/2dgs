conda env create -n instant-gi python=3.12
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install uv
pip install --no-build-isolation -r requirements.txt

pip install --no-build-isolation --force-reinstall --no-cache-dir ./submodules/ellipse_fit
pip install --no-build-isolation --force-reinstall --no-cache-dir ./submodules/torch_dither