"""
Dust3r-based dense depth inference and caching using mini-dust3r.

This module intentionally keeps the Dust3r integration separate so gs_init
stays focused on lifting logic. It expects the mini-dust3r library to provide
an API compatible with the `MiniDUST3R.from_pretrained(...).predict_depths(...)`
pattern. If your installation exposes a different entrypoint, adapt the call
in `compute_aligned_depths`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_ID = "dust3r/mini-dust3r"  # adjust to your preferred checkpoint


def _hash_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def load_dust3r_model(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
    """
    Load a Dust3r model via mini-dust3r. Cached by the caller.
    """
    try:
        from mini_dust3r import MiniDUST3R  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mini-dust3r is required. Install with "
            "`pip install mini-dust3r` or add it to your environment."
        ) from exc

    model = MiniDUST3R.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model


def compute_aligned_depths(
    parser,
    model,
    device: torch.device,
    cache_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute globally aligned depth maps for all images in the parser using Dust3r.

    Returns:
        Dict mapping absolute image path -> depth tensor [1,1,H,W] on CPU.
    """
    depth_maps: Dict[str, torch.Tensor] = {}
    image_paths: List[str] = parser.image_paths

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        key = _hash_path(Path(img_path))
        cached = cache_dir / f"{key}.npy" if cache_dir is not None else None
        if cached is not None and cached.exists():
            depth_np = np.load(cached)
            depth_maps[img_path] = torch.from_numpy(depth_np.astype(np.float32))
            continue

        pil = Image.open(img_path).convert("RGB")
        width, height = pil.size
        # Prepare intrinsics/extrinsics
        idx = parser.image_paths.index(img_path)
        K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[idx]]).float().to(device)
        c2w = torch.from_numpy(parser.camtoworlds[idx]).float().to(device)
        # mini-dust3r is expected to accept a list of images with intrinsics/extrinsics
        if not hasattr(model, "predict_depths"):
            raise RuntimeError(
                "Dust3r model does not expose predict_depths; please adapt compute_aligned_depths."
            )
        depth = model.predict_depths(
            images=[pil],
            intrinsics=[K],
            poses=[c2w],
            device=device,
        )[0]  # expected [H,W]
        depth_t = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Resize to original in case model changes resolution
        if depth_t.shape[-2:] != (height, width):
            depth_t = torch.nn.functional.interpolate(
                depth_t, size=(height, width), mode="bilinear", align_corners=True
            )
        if cached is not None:
            np.save(cached, depth_t.cpu().numpy().astype(np.float32))
        depth_maps[img_path] = depth_t.cpu()

    return depth_maps
