import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

class ProjectedChamferLoss(nn.Module):
    """
    Computes Chamfer distance between projected 2D Gaussians and Ground Truth.
    Output is normalized to [-1, 1] NDC space for numerical stability.
    """
    def __init__(
        self, 
        gaussian_dataset, 
        image_paths: List[str],
        chunk_size: int = 8192
    ):
        super().__init__()
        self.dataset = gaussian_dataset
        self.image_paths = image_paths
        self.chunk_size = chunk_size
        
        # Cache for Ground Truth points to avoid repeated disk I/O
        self._cache: Dict[str, torch.Tensor] = {}
        self._warned_gt_indices: Set[int] = set()
        self._warned_pred_indices: Set[Tuple[int, int]] = set()

    @torch.no_grad()
    def _get_gt_points(
        self, image_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Retrieves and caches ground truth 2D points for a specific image index."""
        if self.dataset is None:
            return None, "Gaussian dataset is unavailable."
        if image_idx < 0 or image_idx >= len(self.image_paths):
            return None, (
                f"Image index {image_idx} out of bounds "
                f"(available={len(self.image_paths)})."
            )

        path = self.image_paths[image_idx]
        stem = Path(path).stem

        if stem not in self._cache:
            # Assumes dataset returns a dict with 'xy' tensor
            try:
                data = self.dataset.get_gaussians_for_image(path)
            except KeyError:
                return None, f"No gaussian data found for '{stem}'."
            if data is None:
                return None, f"Dataset returned None for '{stem}'."
            self._cache[stem] = data["xy"].detach().float()

        points = self._cache[stem]
        if points.numel() == 0:
            return None, f"Ground truth gaussian set empty for '{stem}'."

        return points, None

    def _get_pred_points(
        self,
        info: Dict,
        cam_idx: int,
        screen_size: torch.Tensor,
        packed: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Extracts and normalizes predicted 2D means from render info."""
        means2d = info.get("means2d")
        if means2d is None:
            return None, "render_info missing 'means2d'."

        # Extract points based on packing strategy
        if packed:
            cam_ids = info.get("camera_ids")
            if cam_ids is None:
                return None, "render_info missing 'camera_ids' (packed mode)."
            batch_ids = info.get("batch_ids")
            if batch_ids is None:
                return None, "render_info missing 'batch_ids' (packed mode)."
            mask = ((cam_ids == cam_idx) & (batch_ids == 0))
            if not mask.any().item():
                return None, f"No projected gaussians matched cam_idx={cam_idx}."
            points = means2d[mask]
        else:
            # Unpacked: [Batch, Cam, Points, 2]
            radii = info.get("radii")
            if radii is None:
                return None, "render_info missing 'radii' (unpacked mode)."
            if cam_idx >= radii.shape[1]:
                return None, (
                    f"cam_idx {cam_idx} >= number of cameras {radii.shape[1]} "
                    "in unpacked projection."
                )
            view_radii = radii[0, cam_idx]
            valid_mask = (view_radii > 0).all(dim=-1)
            points = means2d[0, cam_idx][valid_mask]

        if points.numel() == 0:
            return None, f"No visible gaussians for cam_idx={cam_idx}."

        # Normalize Pixel Coordinates -> NDC [-1, 1]
        # (x / w) * 2 - 1
        return points / (screen_size - 1.0) * 2.0 - 1.0, None

    def forward(
        self, 
        render_info: Dict, 
        view_indices: torch.Tensor, 
        width: int, 
        height: int, 
        packed: bool = True
    ) -> torch.Tensor:
        
        if self.dataset is None:
            return torch.tensor(0.0, device=view_indices.device)

        device = view_indices.device
        screen_size = torch.tensor([width, height], device=device)
        losses = []

        for batch_i, view_idx in enumerate(view_indices.tolist()):
            # 1. Get Ground Truth (Target)
            gt_xy, gt_reason = self._get_gt_points(view_idx)
            if gt_xy is None or gt_xy.numel() == 0:
                self._warn_missing_gt(view_idx, gt_reason)
                continue

            # 2. Get Prediction (Source) -- assume single camera view.
            pred_xy, pred_reason = self._get_pred_points(
                render_info, batch_i, screen_size, packed
            )
            if pred_xy is None:
                self._warn_missing_pred(view_idx, batch_i, pred_reason)
                continue

            # 3. Compute Distance
            loss = symmetric_chamfer_distance(
                gt_xy.to(device), pred_xy, self.chunk_size
            )
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device)
            
        return torch.stack(losses).mean()

    def _warn_missing_gt(self, view_idx: int, reason: Optional[str]) -> None:
        if view_idx in self._warned_gt_indices:
            return
        self._warned_gt_indices.add(view_idx)
        detail = reason or "unknown reason."
        hint = (
            self.image_paths[view_idx]
            if 0 <= view_idx < len(self.image_paths)
            else "unknown path"
        )
        warnings.warn(
            f"[ProjectedChamferLoss] No GT gaussians for view {view_idx} ({hint}): {detail}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _warn_missing_pred(
        self, view_idx: int, cam_idx: int, reason: Optional[str]
    ) -> None:
        key = (view_idx, cam_idx)
        if key in self._warned_pred_indices:
            return
        self._warned_pred_indices.add(key)
        detail = reason or "unknown reason."
        hint = (
            self.image_paths[view_idx]
            if 0 <= view_idx < len(self.image_paths)
            else "unknown path"
        )
        warnings.warn(
            f"[ProjectedChamferLoss] No predicted gaussians for cam {cam_idx} / view {view_idx} ({hint}): {detail}",
            RuntimeWarning,
            stacklevel=3,
        )


# --- Math Utilities (Functional) ---

def symmetric_chamfer_distance(
    x: torch.Tensor, y: torch.Tensor, chunk_size: int = 8192
) -> torch.Tensor:
    """Computes symmetric Chamfer distance: mean(min(x,y)) + mean(min(y,x))."""
    x, y = x.float(), y.float()
    
    x_min = _chunked_min_dist(x, y, chunk_size)
    y_min = _chunked_min_dist(y, x, chunk_size)

    if x_min is None or y_min is None:
        return torch.tensor(0.0, device=x.device)

    return x_min.mean() + y_min.mean()

def _chunked_min_dist(
    src: torch.Tensor, tgt: torch.Tensor, chunk_size: int
) -> Optional[torch.Tensor]:
    """Helper: Computes minimum distance from src to tgt in memory-safe chunks."""
    n_src = src.shape[0]
    if n_src == 0 or tgt.numel() == 0:
        return None

    mins = []
    # Loop over src points in chunks
    for i in range(0, n_src, chunk_size):
        end = min(i + chunk_size, n_src)
        # cdist computes pairwise distance matrix [chunk_size, n_tgt]
        dists = torch.cdist(src[i:end], tgt)
        mins.append(dists.min(dim=1).values)

    return torch.cat(mins)
