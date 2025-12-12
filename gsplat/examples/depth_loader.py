"""
Helpers for loading dense depth maps produced by Depth Anything 3.

Depth maps are expected to be saved per scene in a directory with filenames
matching the training images (same stem). The DA3 mip-NeRF processor writes:

    <depth_dir>/<image_stem>.npy   float32 depth in meters
    <depth_dir>/<image_stem>.png   uint16 depth in millimeters (optional)

This module provides a lightweight loader that can:
- Find the corresponding depth file for an image stem.
- Load .npy or .png depths.
- Optionally resize to a target (H, W) to match downsampled training images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image


DepthArray = np.ndarray


@dataclass
class DepthLoader:
    """Load per-image dense depth maps from a directory."""

    depth_dir: Union[str, Path]
    prefer_npy: bool = True
    cache: bool = True
    _depth_cache: Dict[str, DepthArray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.depth_dir = Path(self.depth_dir)

    def depth_path_for_stem(self, stem: str) -> Optional[Path]:
        """Return a depth path for a given image stem, or None if missing."""
        npy = self.depth_dir / f"{stem}.npy"
        png = self.depth_dir / f"{stem}.png"

        if self.prefer_npy:
            if npy.exists():
                return npy
            if png.exists():
                return png
        else:
            if png.exists():
                return png
            if npy.exists():
                return npy
        return None

    def load_depth(
        self, stem: str, target_hw: Optional[Tuple[int, int]] = None
    ) -> Optional[DepthArray]:
        """
        Load depth for an image stem.

        Args:
            stem: Image filename stem.
            target_hw: Optional (H, W) to resize depth to.

        Returns:
            Depth in meters, float32, shape (H, W), or None if missing.
        """
        if self.cache and stem in self._depth_cache:
            depth = self._depth_cache[stem]
            if target_hw is None or depth.shape == target_hw:
                return depth
            return self._resize_depth(depth, target_hw)

        path = self.depth_path_for_stem(stem)
        if path is None:
            return None

        if path.suffix.lower() == ".npy":
            depth = np.load(str(path)).astype(np.float32)
        elif path.suffix.lower() == ".png":
            depth_mm = np.array(Image.open(path), dtype=np.uint16)
            depth = (depth_mm.astype(np.float32) / 1000.0)  # mm -> meters
        else:
            return None

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth = np.clip(depth, 0.0, None)

        if self.cache:
            self._depth_cache[stem] = depth

        if target_hw is not None and depth.shape != target_hw:
            depth = self._resize_depth(depth, target_hw)
        return depth

    @staticmethod
    def _resize_depth(depth: DepthArray, target_hw: Tuple[int, int]) -> DepthArray:
        """Resize a depth map to (H, W) using bilinear interpolation."""
        h_t, w_t = target_hw
        depth_img = Image.fromarray(depth.astype(np.float32))
        depth_img = depth_img.resize((w_t, h_t), resample=Image.BILINEAR)
        return np.array(depth_img, dtype=np.float32)

