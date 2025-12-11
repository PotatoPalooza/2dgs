"""
Depth Anything V2 integration using the transformers library (no extra vendor packages).

Provides:
    - load_depth_model: returns (processor, model) cached in memory.
    - infer_depth_map: runs the model and caches per-image depth maps.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_REPO = "depth-anything/Depth-Anything-V2-Small-hf"

_MODEL_CACHE = {}
_DEPTH_CACHE = {}


def _hash_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def load_depth_model(
    repo_id: str = DEFAULT_REPO,
    device: str = "cpu",
) -> Tuple[object, torch.nn.Module]:
    """
    Load processor + model via transformers; cached in memory.
    """
    key = (repo_id, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForDepthEstimation.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = (processor, model)
    return processor, model


def infer_depth_map(
    image_path: Path,
    processor,
    model: torch.nn.Module,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Run Depth Anything V2 and return a depth map (torch, shape [1,1,H,W]).
    Uses an on-disk cache if cache_dir is provided, otherwise an in-memory cache.
    """
    image_path = Path(image_path)
    cache_key = _hash_path(image_path)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / f"{cache_key}.npy"
        if cached_file.exists():
            depth_np = np.load(cached_file)
            return torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)

    if cache_dir is None and cache_key in _DEPTH_CACHE:
        return _DEPTH_CACHE[cache_key].to(device=device)

    pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pred = model(**inputs)
        depth = pred.predicted_depth  # (1, h, w)
        # Upsample to original image size
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=pil.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=True,
        )

    if cache_dir is not None:
        np.save(cached_file, depth.cpu().numpy().astype(np.float32))
    else:
        _DEPTH_CACHE[cache_key] = depth.cpu()

    return depth
