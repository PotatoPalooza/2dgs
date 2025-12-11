"""
Chamfer loss utilities for comparing 2D Gaussian projections.

The loss is computed in a chunked fashion to keep the pairwise distance
matrix small enough to fit in memory while still running on the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from gaussian_image_dataset import GaussianImageDataset


def _chunked_min_dist(
    src: Tensor, tgt: Tensor, chunk_size: int = 8192
) -> Optional[Tensor]:
    """Return per-point min distance from src->tgt using chunked cdist."""
    if src.numel() == 0 or tgt.numel() == 0:
        return None

    mins = []
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        dists = torch.cdist(src[start:end], tgt)  # [chunk, |tgt|]
        mins.append(dists.min(dim=1).values)
    return torch.cat(mins, dim=0) if mins else None


def batched_chamfer_distance(
    a: Tensor, b: Tensor, chunk_size: int = 8192
) -> Tensor:
    """
    Symmetric Chamfer distance between two 2D point sets.

    Args:
        a: [Na, 2] tensor of source points.
        b: [Nb, 2] tensor of target points.
        chunk_size: Number of source points processed per chunk. Lower this
            if you run into memory issues during the distance computation.
    """
    a = a.float()
    b = b.float()

    a_to_b = _chunked_min_dist(a, b, chunk_size)
    b_to_a = _chunked_min_dist(b, a, chunk_size)

    if a_to_b is None or b_to_a is None:
        # If either set is empty, fall back to zero so we don't NaN the loss.
        device = a.device if a.numel() else b.device
        return torch.zeros((), device=device)

    return a_to_b.mean() + b_to_a.mean()


class ChamferLoss:
    def __init__(
        self,
        gaussian_dataset: Optional[GaussianImageDataset],
        image_to_gaussians: Dict[str, object],
        train_image_paths: Iterable[str],
        device: torch.device,
        packed: bool,
        chunk_size: int = 8192,
    ) -> None:
        self.gaussian_dataset = gaussian_dataset
        self.image_to_gaussians = image_to_gaussians
        self.train_image_paths = list(train_image_paths)
        self.device = device
        self.packed = packed
        self.chunk_size = chunk_size
        self._frame_cache: Dict[str, Tensor] = {}

    def _get_gaussian_xy(self, image_path: str) -> Optional[Tensor]:
        if self.gaussian_dataset is None:
            return None
        stem = Path(image_path).stem
        if stem not in self.image_to_gaussians:
            return None
        if stem not in self._frame_cache:
            params = self.gaussian_dataset.get_gaussians_for_image(image_path)
            self._frame_cache[stem] = params["xy"].detach().float()
        return self._frame_cache[stem]

    def _projected_gaussian_means2d(
        self, info: Dict, cam_idx: int, width: int, height: int
    ) -> Optional[Tensor]:
        if self.packed:
            camera_ids = info.get("camera_ids")
            means2d = info.get("means2d")
            batch_ids = info.get("batch_ids")
            if camera_ids is None or means2d is None:
                return None
            mask = camera_ids == cam_idx
            if batch_ids is not None:
                mask = mask & (batch_ids == 0)
            proj = means2d[mask]
        else:
            means2d = info.get("means2d")
            radii = info.get("radii")
            if means2d is None or radii is None:
                return None
            proj = means2d[0, cam_idx]
            valid = (radii[0, cam_idx] > 0).all(dim=-1)
            proj = proj[valid]
        if proj.numel() == 0:
            return None
        scale = torch.tensor([width - 1.0, height - 1.0], device=proj.device)
        return proj / scale * 2.0 - 1.0

    def __call__(
        self, info: Dict, image_ids: Tensor, width: int, height: int
    ) -> Tensor:
        if self.gaussian_dataset is None:
            return torch.zeros((), device=self.device)
        losses = []
        for cam_idx, dataset_idx in enumerate(image_ids.tolist()):
            if dataset_idx >= len(self.train_image_paths):
                continue
            image_path = self.train_image_paths[dataset_idx]
            target_xy = self._get_gaussian_xy(image_path)
            if target_xy is None or target_xy.numel() == 0:
                continue
            proj_xy = self._projected_gaussian_means2d(info, cam_idx, width, height)
            if proj_xy is None or proj_xy.numel() == 0:
                continue
            losses.append(
                batched_chamfer_distance(
                    target_xy.to(self.device),
                    proj_xy,
                    chunk_size=self.chunk_size,
                )
            )
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()
