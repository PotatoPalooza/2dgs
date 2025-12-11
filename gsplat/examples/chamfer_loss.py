import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path

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

    @torch.no_grad()
    def _get_gt_points(self, image_idx: int) -> Optional[torch.Tensor]:
        """Retrieves and caches ground truth 2D points for a specific image index."""
        if self.dataset is None or image_idx >= len(self.image_paths):
            return None

        path = self.image_paths[image_idx]
        stem = Path(path).stem

        if stem not in self._cache:
            # Assumes dataset returns a dict with 'xy' tensor
            data = self.dataset.get_gaussians_for_image(path)
            if data is None: 
                return None
            self._cache[stem] = data["xy"].detach().float()
            
        return self._cache[stem]

    def _get_pred_points(
        self, 
        info: Dict, 
        batch_idx: int,
        cam_idx: int, 
        screen_size: torch.Tensor, 
        packed: bool
    ) -> Optional[torch.Tensor]:
        """Extracts and normalizes predicted 2D means from render info."""
        means2d = info.get("means2d")
        if means2d is None:
            return None

        # Extract points based on packing strategy
        if packed:
            cam_ids = info.get("camera_ids")
            if cam_ids is None:
                return None
            batch_ids = info.get("batch_ids")
            if batch_ids is None:
                batch_ids = torch.zeros_like(cam_ids)
            mask = (cam_ids == cam_idx) & (batch_ids == batch_idx)
            points = means2d[mask]
        else:
            # Unpacked: [Batch, Cam, Points, 2]
            radii = info.get("radii")
            if radii is None: return None
            if batch_idx >= means2d.shape[0]:
                return None
            valid_mask = (radii[batch_idx, cam_idx] > 0).all(dim=-1)
            points = means2d[batch_idx, cam_idx][valid_mask]

        if points.numel() == 0:
            return None

        # Normalize Pixel Coordinates -> NDC [-1, 1]
        # (x / w) * 2 - 1
        return points / (screen_size - 1.0) * 2.0 - 1.0

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
            gt_xy = self._get_gt_points(view_idx)
            if gt_xy is None or gt_xy.numel() == 0:
                continue

            # 2. Get Prediction (Source). For our dataloader each sample has a single view (cam_idx=0).
            pred_xy = self._get_pred_points(render_info, batch_i, 0, screen_size, packed)
            if pred_xy is None:
                continue

            # 3. Compute Distance
            loss = symmetric_chamfer_distance(
                gt_xy.to(device), pred_xy, self.chunk_size
            )
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device)
            
        return torch.stack(losses).mean()


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
