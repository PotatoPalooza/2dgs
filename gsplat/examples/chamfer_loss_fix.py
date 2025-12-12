import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Literal
from pathlib import Path

# --- Utility Functions (Kept as is) ---

def symmetric_chamfer_distance(
    x: torch.Tensor, y: torch.Tensor, chunk_size: int = 8192
) -> torch.Tensor:
    """
    Computes symmetric Chamfer distance: mean(min(x,y)) + mean(min(y,x)).
    
    Args:
        x: [N, D] feature vectors (ground truth)
        y: [M, D] feature vectors (predictions)
        chunk_size: Process in chunks for memory efficiency
        
    Returns:
        Scalar distance
    """
    x, y = x.float(), y.float()
    
    x_min = _chunked_min_dist(x, y, chunk_size)
    y_min = _chunked_min_dist(y, x, chunk_size)

    if x_min is None or y_min is None:
        return torch.tensor(0.0, device=x.device)

    return x_min.mean() + y_min.mean()


def _chunked_min_dist(
    src: torch.Tensor, tgt: torch.Tensor, chunk_size: int
) -> Optional[torch.Tensor]:
    """
    Helper: Computes minimum distance from src to tgt in memory-safe chunks.
    """
    n_src = src.shape[0]
    if n_src == 0 or tgt.numel() == 0:
        return None

    mins = []
    # Loop over src points in chunks
    for i in range(0, n_src, chunk_size):
        end = min(i + chunk_size, n_src)
        # cdist computes pairwise Euclidean distance matrix [chunk_size, n_tgt]
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
    """Project 3D points into image space with gradients."""
    if points.numel() == 0:
        return None
    ones = torch.ones_like(points[:, :1])
    homo = torch.cat([points, ones], dim=1)  # [N,4]
    cam = homo @ viewmat.T  # [N,4]
    cam = cam[:, :3]
    depth = cam[:, 2:3]
    pos = depth.squeeze(-1) > 1e-6
    if not pos.any():
        return None
    cam = cam[pos]
    pixels = cam @ K.T
    xy = pixels[:, :2] / pixels[:, 2:3].clamp_min(1e-6)
    scale = torch.tensor([width - 1.0, height - 1.0], device=xy.device, dtype=xy.dtype)
    return xy / scale * 2.0 - 1.0


def farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for more uniform coverage.
    More expensive but gives better coverage than random sampling.
    
    Args:
        points: [N, D] point features
        n_samples: Number of points to sample
        
    Returns:
        Indices of sampled points [n_samples]
    """
    N, D = points.shape
    if N <= n_samples:
        return torch.arange(N, device=points.device)
    
    # Initialize with random point
    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    sampled_indices[0] = torch.randint(0, N, (1,), device=points.device)
    
    # Distance to nearest sampled point
    distances = torch.full((N,), float('inf'), device=points.device)
    
    for i in range(1, n_samples):
        # Update distances to nearest sampled point
        last_point = points[sampled_indices[i-1]]
        new_dists = torch.norm(points - last_point.unsqueeze(0), dim=1)
        distances = torch.min(distances, new_dists)
        
        # Sample farthest point
        sampled_indices[i] = torch.argmax(distances)
    
    return sampled_indices

# --- ProjectedChamferLoss Class (Modified) ---

class ProjectedChamferLoss(nn.Module):
    """
    Computes Chamfer distance between projected 2D Gaussians and Ground Truth
    using all Gaussian attributes with optional point subsampling.
    """
    def __init__(
        self, 
        gaussian_dataset, 
        image_paths: List[str],
        chunk_size: int = 8192,
        attribute_mode: Literal["xy_only", "all", "geometric"] = "all",
        weights: Optional[Dict[str, float]] = None,
        max_points: Optional[int] = 10000,  # NEW: Maximum points to use
        sampling_mode: Literal["random", "fps"] = "random",  # NEW: Sampling strategy
    ):
        super().__init__()
        self.dataset = gaussian_dataset
        self.image_paths = image_paths
        self.chunk_size = chunk_size
        self.attribute_mode = attribute_mode
        self.max_points = max_points  # NEW: e.g., 10000
        self.sampling_mode = sampling_mode  # NEW
        
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _choose_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        Pick a shared set of indices for all attributes based on xy positions.
        """
        n_points = points.shape[0]
        if self.max_points is None or n_points <= self.max_points:
            return torch.arange(n_points, device=points.device)

        if self.sampling_mode == "fps":
            return farthest_point_sampling(points, self.max_points)
        if self.sampling_mode == "random":
            return torch.randperm(n_points, device=points.device)[: self.max_points]

        raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def _subsample_gaussian_dict(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the same subsampling indices across all Gaussian attributes.
        """
        xy = data.get("xy")
        if xy is None or xy.numel() == 0:
            return data

        idx = self._choose_indices(xy)
        return {k: v[idx] for k, v in data.items() if v is not None}

    @torch.no_grad()
    def _get_gt_gaussians(self, image_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves and caches ground truth 2D Gaussian parameters."""
        if self.dataset is None or image_idx >= len(self.image_paths):
            return None

        path = self.image_paths[image_idx]
        stem = Path(path).stem

        if stem not in self._cache:
            data = self.dataset.get_gaussians_for_image(path)
            if data is None: 
                return None
            
            # Store all attributes - Expected shape: [N, D]
            self._cache[stem] = {
                "xy": data["xy"].detach().float(),      # [N, 2]
                "scale": data["scale"].detach().float(),    # [N, 2]
                "rotation": data["rotation"].detach().float(), # [N, 1]
                "opacity": data["opacity"].detach().float(), # [N, 1]
                "color": data["color"].detach().float(),    # [N, 3]
            }
            
        return self._cache[stem]

    def _get_pred_gaussians(
        self,
        info: Dict,
        batch_idx: int,
        packed: bool,
        gaussian_means: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        gaussian_colors: Optional[torch.Tensor],
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Project predicted 3D Gaussians to 2D with gradients. Supports packed and unpacked.
        """
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
            mask = (batch_ids == batch_idx) & (camera_ids == 0)
            if not mask.any():
                return None
            ids = gaussian_ids[mask].long()
        else:
            radii = info.get("radii")
            if radii is None:
                return None
            if radii.dim() == 3:
                # [C, N, 2]
                if batch_idx >= radii.shape[0]:
                    return None
                view_radii = radii[batch_idx]
            else:
                # [B, C, N, 2]
                if batch_idx >= radii.shape[0]:
                    return None
                view_radii = radii[batch_idx, 0]
            valid_mask = (view_radii > 0).all(dim=-1)
            if not valid_mask.any():
                return None
            ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).long()

        points3d = gaussian_means[ids]
        viewmat = viewmats[min(batch_idx, viewmats.shape[0] - 1)]
        K = Ks[min(batch_idx, Ks.shape[0] - 1)]
        xy = _project_points(points3d, viewmat, K, width, height)
        if xy is None or xy.numel() == 0:
            return None

        result = {"xy": xy}
        # Use differentiable parameters directly
        result["opacity"] = torch.sigmoid(gaussian_opacities[ids]).unsqueeze(-1)  # [N,1]
        if gaussian_colors is not None:
            result["color"] = gaussian_colors[ids]  # [N,3]

        return result

    def forward(
        self, 
        render_info: Dict, 
        view_indices: torch.Tensor, 
        width: int, 
        height: int, 
        packed: bool = False,
        camtoworlds: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        gaussian_means: Optional[torch.Tensor] = None,
        gaussian_scales: Optional[torch.Tensor] = None,
        gaussian_opacities: Optional[torch.Tensor] = None,
        gaussian_colors: Optional[torch.Tensor] = None,
        step: int = 0,
        use_attr_loss: bool = True,
        color_warmup: int = 500,
        attr_weight: float = 0.1,
    ) -> torch.Tensor:
        
        if self.dataset is None:
            return torch.tensor(0.0, device=view_indices.device)
        if (
            camtoworlds is None
            or Ks is None
            or gaussian_means is None
            or gaussian_scales is None
            or gaussian_opacities is None
        ):
            raise ValueError(
                "Chamfer loss needs camtoworlds, Ks, gaussian_means, scales, opacities."
            )

        device = view_indices.device
        camtoworlds = camtoworlds.to(device)
        Ks = Ks.to(device)
        gaussian_means = gaussian_means.to(device)
        gaussian_scales = gaussian_scales.to(device)
        gaussian_opacities = gaussian_opacities.to(device)
        gaussian_colors = gaussian_colors.to(device) if gaussian_colors is not None else None
        viewmats = torch.linalg.inv(camtoworlds)
        losses = []

        # Determine which attributes should contribute based on mode and warmup
        if not use_attr_loss:
            attr_keys: List[str] = []
        else:
            if self.attribute_mode == "xy_only":
                attr_keys = []
            elif self.attribute_mode == "geometric":
                attr_keys = []
            elif self.attribute_mode == "all":
                attr_keys = ["opacity"]
                if step >= color_warmup:
                    attr_keys.append("color")
            else:
                raise ValueError(f"Unknown attribute_mode {self.attribute_mode}")

        if "color" in attr_keys and gaussian_colors is None:
            raise ValueError("attribute_mode requires color supervision but gaussian_colors is None.")

        for batch_i, view_idx in enumerate(view_indices.tolist()):
            # 1. Get Ground Truth (Target) - Expected shape: [N, D_f]
            gt_data = self._get_gt_gaussians(view_idx)
            if gt_data is None:
                continue

            # 2. Get Prediction (Source) - Expected shape: [N, D_f]
            pred_data = self._get_pred_gaussians(
                render_info,
                batch_i,
                packed,
                gaussian_means,
                gaussian_scales,
                gaussian_opacities,
                gaussian_colors,
                viewmats,
                Ks,
                width,
                height,
            )
            if pred_data is None:
                continue

            # Move GT tensors to device before subsampling so indices align
            gt_data = {k: v.to(device) for k, v in gt_data.items()}

            # Subsample consistently across attributes
            gt_data = self._subsample_gaussian_dict(gt_data)
            pred_data = self._subsample_gaussian_dict(pred_data)

            gt_xy = gt_data["xy"]
            pred_xy = pred_data["xy"]

            if gt_xy.numel() == 0 or pred_xy.numel() == 0:
                continue

            chamfer_xy = symmetric_chamfer_distance(
                gt_xy, pred_xy, self.chunk_size
            )

            # NN matching from xy to regress other attrs
            attr_loss = torch.tensor(0.0, device=device)
            dists = torch.cdist(pred_xy.detach(), gt_xy.detach())  # [Np, Ng]
            nn_pred_to_gt = dists.argmin(dim=1)  # [Np]
            nn_gt_to_pred = dists.argmin(dim=0)  # [Ng]

            for key in attr_keys:
                if key not in pred_data or key not in gt_data:
                    continue

                pred_attr = pred_data[key]
                gt_attr = gt_data[key]
                if pred_attr.shape[0] != pred_xy.shape[0] or gt_attr.shape[0] != gt_xy.shape[0]:
                    continue

                matched_gt = gt_attr[nn_pred_to_gt]
                matched_pred = pred_attr[nn_gt_to_pred]
                attr_loss = attr_loss + F.l1_loss(pred_attr, matched_gt)
                attr_loss = attr_loss + F.l1_loss(gt_attr, matched_pred)

            total = chamfer_xy + attr_weight * attr_loss
            losses.append(total)

        if not losses:
            return torch.tensor(0.0, device=device)
            
        return torch.stack(losses).mean()
