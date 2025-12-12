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


def _conic_to_scale_rot(conics: torch.Tensor) -> Optional[tuple]:
    if conics is None or conics.numel() == 0:
        return None
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    trace = a + c
    det = a * c - b * b
    discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
    sqrt_disc = torch.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    scale_x = torch.clamp(1.0 / torch.sqrt(torch.clamp(lambda1, min=1e-6)), max=10.0)
    scale_y = torch.clamp(1.0 / torch.sqrt(torch.clamp(lambda2, min=1e-6)), max=10.0)
    rotation = torch.atan2(b, lambda1 - a).unsqueeze(-1) % (2 * 3.14159265)
    return torch.stack([scale_x, scale_y], dim=-1), rotation

# --- NEW: Subsampling utility ---

def random_subsample(tensor: torch.Tensor, max_points: int, dim: int = 0) -> torch.Tensor:
    """
    Randomly subsample tensor along specified dimension.
    
    Args:
        tensor: Input tensor to subsample
        max_points: Maximum number of points to keep
        dim: Dimension to subsample along (default: 0)
        
    Returns:
        Subsampled tensor
    """
    n_points = tensor.shape[dim]
    if n_points <= max_points:
        return tensor
    
    # Random indices without replacement
    indices = torch.randperm(n_points, device=tensor.device)[:max_points]
    
    # Index along the specified dimension
    if dim == 0:
        return tensor[indices]
    elif dim == 1:
        return tensor[:, indices]
    else:
        # For higher dimensions, use advanced indexing
        index_tuple = [slice(None)] * tensor.ndim
        index_tuple[dim] = indices
        return tensor[tuple(index_tuple)]


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
        attribute_mode: Literal["xy_only", "all", "geometric"] = "geometric",
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
        
        default_weights = {
            "xy": 1.0, 
            "scale": 0.5, 
            "rotation": 0.3, 
            "opacity": 0.2, 
            "color": 0.5 
        }
        self.weights = weights if weights is not None else default_weights
        
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._scale_norm: Optional[float] = None
        self._rotation_norm: Optional[float] = None

    def _subsample_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Subsample features to max_points if specified.
        
        Args:
            features: [N, D] feature tensor
            
        Returns:
            Subsampled features [min(N, max_points), D]
        """
        if self.max_points is None or features.shape[0] <= self.max_points:
            return features
        
        if self.sampling_mode == "random":
            return random_subsample(features, self.max_points, dim=0)
        elif self.sampling_mode == "fps":
            indices = farthest_point_sampling(features, self.max_points)
            return features[indices]
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

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

    def _normalize_attributes(
        self,
        xy: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        rotation: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        color: Optional[torch.Tensor] = None,
        attr_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Normalize and concatenate attributes into a feature vector [N, D_f]."""
        mode = attr_mode if attr_mode is not None else self.attribute_mode
        features = [xy]

        if mode == "xy_only":
            return torch.cat(features, dim=-1)

        if mode in ["geometric", "all"]:
            if scale is not None:
                if self._scale_norm is None:
                    self._scale_norm = 10.0
                scale_norm = (scale / self._scale_norm) * self.weights["scale"]
                features.append(scale_norm)

            if rotation is not None:
                if self._rotation_norm is None:
                    self._rotation_norm = 2 * 3.14159265
                rotation_norm = (rotation / self._rotation_norm) * self.weights["rotation"]
                features.append(rotation_norm)

        if mode == "all":
            if opacity is not None:
                opacity_norm = opacity * self.weights["opacity"]
                features.append(opacity_norm)

            if color is not None:
                color_norm = color * self.weights["color"]
                features.append(color_norm)

        return torch.cat(features, dim=-1)

    def _get_pred_gaussians(
        self,
        info: Dict,
        batch_idx: int,
        screen_size: torch.Tensor,
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
        xy_norm = xy / (screen_size - 1.0) * 2.0 - 1.0

        result = {"xy": xy_norm}
        # Use differentiable parameters directly
        result["scale"] = torch.exp(gaussian_scales[ids])[:, :2]  # [N,2]
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
    ) -> torch.Tensor:
        
        if self.dataset is None:
            return torch.tensor(0.0, device=view_indices.device)
        if (
            camtoworlds is None
            or Ks is None
            or gaussian_means is None
            or gaussian_scales is None
            or gaussian_opacities is None
            or gaussian_colors is None
        ):
            raise ValueError(
                "Chamfer loss needs camtoworlds, Ks, gaussian_means, scales, opacities, colors."
            )

        device = view_indices.device
        screen_size = torch.tensor([width, height], device=device, dtype=torch.float32)
        camtoworlds = camtoworlds.to(device)
        Ks = Ks.to(device)
        gaussian_means = gaussian_means.to(device)
        gaussian_scales = gaussian_scales.to(device)
        gaussian_opacities = gaussian_opacities.to(device)
        gaussian_colors = gaussian_colors.to(device)
        viewmats = torch.linalg.inv(camtoworlds)
        losses = []

        for batch_i, view_idx in enumerate(view_indices.tolist()):
            # 1. Get Ground Truth (Target) - Expected shape: [N, D_f]
            gt_data = self._get_gt_gaussians(view_idx)
            if gt_data is None:
                continue

            # 2. Get Prediction (Source) - Expected shape: [N, D_f]
            pred_data = self._get_pred_gaussians(
                render_info,
                batch_i,
                screen_size,
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
            
            # Subsample xy consistently
            gt_xy = self._subsample_features(gt_data["xy"].to(device))
            pred_xy = self._subsample_features(pred_data["xy"])

            if gt_xy.numel() == 0 or pred_xy.numel() == 0:
                continue

            chamfer_xy = symmetric_chamfer_distance(
                gt_xy, pred_xy, self.chunk_size
            )

            # NN matching from xy to regress other attrs
            attr_loss = torch.tensor(0.0, device=device)
            dists = torch.cdist(pred_xy, gt_xy)  # [Np, Ng]
            nn_idx = dists.argmin(dim=1)  # [Np]

            if "opacity" in pred_data and "opacity" in gt_data:
                gt_op = self._subsample_features(gt_data["opacity"].to(device))
                pred_op = self._subsample_features(pred_data["opacity"])
                if gt_op.shape[0] == gt_xy.shape[0] and pred_op.shape[0] == pred_xy.shape[0]:
                    matched = gt_op[nn_idx]
                    attr_loss = attr_loss + torch.nn.functional.l1_loss(pred_op, matched)

            if "scale" in pred_data and "scale" in gt_data:
                gt_scale = self._subsample_features(gt_data["scale"].to(device))
                pred_scale = self._subsample_features(pred_data["scale"])
                if gt_scale.shape[0] == gt_xy.shape[0] and pred_scale.shape[0] == pred_xy.shape[0]:
                    matched = gt_scale[nn_idx]
                    attr_loss = attr_loss + torch.nn.functional.l1_loss(pred_scale, matched)

            if "color" in pred_data and "color" in gt_data:
                gt_color = self._subsample_features(gt_data["color"].to(device))
                pred_color = self._subsample_features(pred_data["color"])
                if gt_color.shape[0] == gt_xy.shape[0] and pred_color.shape[0] == pred_xy.shape[0]:
                    matched = gt_color[nn_idx]
                    attr_loss = attr_loss + torch.nn.functional.l1_loss(pred_color, matched)

            total = chamfer_xy + attr_loss
            losses.append(total)

        if not losses:
            return torch.tensor(0.0, device=device)
            
        return torch.stack(losses).mean()
