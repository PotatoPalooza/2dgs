from pathlib import Path
from typing import Dict, List, Optional

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
        chunk_size: int = 8192,
        max_points: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = gaussian_dataset
        self.image_paths = image_paths
        self.chunk_size = chunk_size
        self.max_points = max_points
        self._debug_prints = 0
        self._debug_print_limit = 10
        
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
        gaussian_means: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        slot_batch: int,
        slot_cam: int,
        width: int,
        height: int,
        packed: bool,
    ) -> Optional[torch.Tensor]:
        """Extracts projected 2D points with gradients back to 3D means."""
        if packed:
            gaussian_ids = info.get("gaussian_ids")
            batch_ids = info.get("batch_ids")
            camera_ids = info.get("camera_ids")
            if (
                gaussian_ids is None
                or batch_ids is None
                or camera_ids is None
                or gaussian_ids.numel() == 0
            ):
                return None
            mask = (batch_ids == slot_batch) & (camera_ids == slot_cam)
            if not mask.any():
                return None
            ids = gaussian_ids[mask].long()
        else:
            radii = info.get("radii")
            if radii is None:
                return None
            if radii.dim() == 3:
                # Shape: [C, N, 2]
                if slot_cam >= radii.shape[0]:
                    return None
                view_radii = radii[slot_cam]
                points2d = info.get("means2d")
                if points2d is None or slot_cam >= points2d.shape[0]:
                    return None
                points2d = points2d[slot_cam]
            else:
                # Shape: [B, C, N, 2]
                if slot_batch >= radii.shape[0] or slot_cam >= radii.shape[1]:
                    return None
                view_radii = radii[slot_batch, slot_cam]
                points2d = info.get("means2d")
                if points2d is None or slot_batch >= points2d.shape[0] or slot_cam >= points2d.shape[1]:
                    return None
                points2d = points2d[slot_batch, slot_cam]
            valid_mask = (view_radii > 0).all(dim=-1)
            if not valid_mask.any():
                return None
            ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).long()

        points3d = gaussian_means[ids]
        viewmat = viewmats[min(slot_batch, viewmats.shape[0] - 1)]
        K = Ks[min(slot_cam, Ks.shape[0] - 1)]
        return _project_points(points3d, viewmat, K, width, height)

    def forward(
        self, 
        render_info: Dict, 
        view_indices: torch.Tensor, 
        width: int, 
        height: int, 
        packed: bool = True,
        camtoworlds: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        gaussian_means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if self.dataset is None:
            return torch.tensor(0.0, device=view_indices.device)
        if camtoworlds is None or Ks is None or gaussian_means is None:
            raise ValueError("Chamfer loss needs camtoworlds, Ks, and gaussian_means for gradients.")

        device = view_indices.device
        camtoworlds = camtoworlds.to(device)
        Ks = Ks.to(device)
        gaussian_means = gaussian_means.to(device)
        viewmats = torch.linalg.inv(camtoworlds)
        losses = []
        if view_indices.numel() == 0:
            return torch.tensor(0.0, device=device)

        for slot, view_idx in enumerate(view_indices.tolist()):
            # 1. Get Ground Truth (Target)
            gt_xy = self._get_gt_points(view_idx)
            if gt_xy is None or gt_xy.numel() == 0:
                if self._debug_prints < self._debug_print_limit:
                    print(f"[ChamferDebug] skip view {view_idx}: missing/empty GT")
                    self._debug_prints += 1
                continue

            # 2. Get Prediction (Source) -- assume single camera view.
            pred_xy = self._get_pred_points(
                info=render_info,
                gaussian_means=gaussian_means,
                viewmats=viewmats,
                Ks=Ks,
                slot_batch=slot,
                slot_cam=0,
                width=width,
                height=height,
                packed=packed,
            )
            if pred_xy is None:
                if self._debug_prints < self._debug_print_limit:
                    print(
                        f"[ChamferDebug] skip view {view_idx} (slot {slot}): "
                        f"no predicted gaussians (packed={packed})"
                    )
                    self._debug_prints += 1
                continue

            # 3. Compute Distance
            gt_xy = _maybe_subsample_points(gt_xy, self.max_points)
            pred_xy = _maybe_subsample_points(pred_xy, self.max_points)
            loss = symmetric_chamfer_distance(
                gt_xy.to(device), pred_xy, self.chunk_size
            )
            losses.append(loss)

        if not losses:
            if self._debug_prints < self._debug_print_limit:
                print(
                    f"[ChamferDebug] no valid chamfer pairs this batch "
                    f"(views={view_indices.tolist()})"
                )
                self._debug_prints += 1
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


def _project_points(
    points: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> Optional[torch.Tensor]:
    if points.numel() == 0:
        return None
    ones = torch.ones_like(points[:, :1])
    homo = torch.cat([points, ones], dim=1)  # [M,4]
    cam = homo @ viewmat.T  # [M,4]
    cam = cam[:, :3]
    depth = cam[:, 2:3]
    pos_mask = depth.squeeze(-1) > 1e-6
    if not pos_mask.any():
        return None
    cam = cam[pos_mask]
    pixels = cam @ K.T
    xy = pixels[:, :2] / pixels[:, 2:3].clamp_min(1e-6)
    scale = torch.tensor([width - 1.0, height - 1.0], device=xy.device, dtype=xy.dtype)
    return xy / scale * 2.0 - 1.0


def _maybe_subsample_points(
    points: torch.Tensor, max_points: Optional[int]
) -> torch.Tensor:
    if max_points is None or points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]
