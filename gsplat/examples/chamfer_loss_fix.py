import torch
import torch.nn as nn
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
        color: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalize and concatenate attributes into a feature vector [N, D_f]."""
        features = [xy]
        
        if self.attribute_mode == "xy_only":
            return torch.cat(features, dim=-1)
        
        # Add geometric attributes
        if self.attribute_mode in ["geometric", "all"]:
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
        
        # Add appearance attributes
        if self.attribute_mode == "all":
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
        packed: bool
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extracts predicted 2D Gaussian attributes from render info.
        Corrected indexing for unpacked tensors with shape [B, N, D] or [B, C, N, D]
        where B=1 and C=1.
        """
        means2d = info.get("means2d")
        if means2d is None:
            return None
        
        if packed:
            # Packed mode uses a 1D mask over a flat tensor [Total_Points, D]
            cam_ids = info.get("camera_ids")
            if cam_ids is None: return None
            batch_ids = info.get("batch_ids", torch.zeros_like(cam_ids))
            
            mask = (cam_ids == batch_idx) & (batch_ids == 0) 
            
            xy = means2d[mask]
            
            conics_flat = info.get("conics")
            conics = conics_flat[mask] if conics_flat is not None else None
            
            opacities_2d_flat = info.get("opacities")
            opacities_2d = opacities_2d_flat[mask] if opacities_2d_flat is not None else None
            
        else:
            # Unpacked mode with shape [1, 138k, D]
            try:
                means2d_slice = means2d[batch_idx] 
            except IndexError:
                means2d_slice = means2d[batch_idx, 0]

            radii = info.get("radii")
            if radii is None: return None

            try:
                radii_slice = radii[batch_idx]
            except IndexError:
                radii_slice = radii[batch_idx, 0]
                
            # MASK FIX: Must use .all(dim=-1) to collapse [N, 2] mask to [N]
            valid_mask = (radii_slice > 0).all(dim=-1)
            
            xy = means2d_slice[valid_mask]
            
            # Apply mask to other attributes
            conics = info.get("conics")
            if conics is not None:
                try:
                    conics_slice = conics[batch_idx]
                except IndexError:
                    conics_slice = conics[batch_idx, 0]

                conics = conics_slice[valid_mask]
                
            opacities_2d = info.get("opacities")
            if opacities_2d is not None:
                try:
                    opacities_2d_slice = opacities_2d[batch_idx]
                except IndexError:
                    opacities_2d_slice = opacities_2d[batch_idx, 0]
                    
                opacities_2d = opacities_2d_slice[valid_mask]

        if xy.numel() == 0:
            return None

        # Normalize xy to NDC [-1, 1]
        xy_norm = xy / (screen_size - 1.0) * 2.0 - 1.0
        
        result = {"xy": xy_norm}
        
        if conics is not None and self.attribute_mode in ["geometric", "all"]:
            # Decompose conic to get scale-like features
            a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
            
            # Compute eigenvalues
            trace = a + c
            det = a * c - b * b
            discriminant = torch.clamp(trace * trace - 4 * det, min=0.0)
            sqrt_disc = torch.sqrt(discriminant)
            
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
            
            scale_x = torch.clamp(1.0 / torch.sqrt(torch.clamp(lambda1, min=1e-6)), max=10.0)
            scale_y = torch.clamp(1.0 / torch.sqrt(torch.clamp(lambda2, min=1e-6)), max=10.0)
            
            result["scale"] = torch.stack([scale_x, scale_y], dim=-1)
            
            # Compute rotation
            vec_x = b
            vec_y = lambda1 - a
            rotation = torch.atan2(vec_y, vec_x).unsqueeze(-1) 
            rotation = rotation % (2 * 3.14159265)
            result["rotation"] = rotation
        
        if opacities_2d is not None and self.attribute_mode == "all":
            result["opacity"] = opacities_2d.unsqueeze(-1) 
        
        return result

    def forward(
        self, 
        render_info: Dict, 
        view_indices: torch.Tensor, 
        width: int, 
        height: int, 
        packed: bool = False
    ) -> torch.Tensor:
        
        if self.dataset is None:
            return torch.tensor(0.0, device=view_indices.device)

        device = view_indices.device
        screen_size = torch.tensor([width, height], device=device, dtype=torch.float32)
        losses = []

        for batch_i, view_idx in enumerate(view_indices.tolist()):
            # 1. Get Ground Truth (Target) - Expected shape: [N, D_f]
            gt_data = self._get_gt_gaussians(view_idx)
            if gt_data is None:
                continue

            # 2. Get Prediction (Source) - Expected shape: [N, D_f]
            pred_data = self._get_pred_gaussians(render_info, batch_i, screen_size, packed)
            if pred_data is None:
                continue
            
            # 3. Normalize and create feature vectors
            gt_features = self._normalize_attributes(
                xy=gt_data["xy"],
                scale=gt_data.get("scale"),
                rotation=gt_data.get("rotation"),
                opacity=gt_data.get("opacity"),
                color=gt_data.get("color")
            ).to(device)
            
            pred_features = self._normalize_attributes(
                xy=pred_data["xy"],
                scale=pred_data.get("scale"),
                rotation=pred_data.get("rotation"),
                opacity=pred_data.get("opacity"),
                color=None 
            )

            if gt_features.numel() == 0 or pred_features.numel() == 0:
                print(f"Empty Chamfer: GT size {gt_features.numel()}, Pred size {pred_features.numel()}")
                continue
            
            # NEW: Subsample both GT and Pred features
            gt_features_sub = self._subsample_features(gt_features)
            pred_features_sub = self._subsample_features(pred_features)
                
            print(f"Chamfer Loss Input Shapes (after subsampling): "
                  f"GT {gt_features.shape} -> {gt_features_sub.shape}, "
                  f"Pred {pred_features.shape} -> {pred_features_sub.shape}")

            # 4. Compute Chamfer Distance on subsampled points
            loss = symmetric_chamfer_distance(
                gt_features_sub, pred_features_sub, self.chunk_size
            )
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device)
            
        return torch.stack(losses).mean()