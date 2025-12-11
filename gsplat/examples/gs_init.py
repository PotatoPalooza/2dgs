"""
Lift 2D Instant-GI gaussians into 3D initialization points.

We project COLMAP sparse points into each training image, then for every 2D
gaussian center we pick the nearest projected COLMAP point in pixel space and
use that 3D position (plus the gaussian RGB) to seed a 3D gaussian. This allows
an initialization that mirrors the 2D gaussian coverage instead of sampling the
COLMAP cloud directly.

The heavy lifting (distance search) is done with torch and chunked so we can
handle ~10M gaussians without blowing memory. Pass a CUDA device to run the
nearest-neighbor search on GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

from datasets.colmap import Parser
from gaussian_image_dataset import GaussianFrame, GaussianImageDataset
from depth_anything import infer_depth_map, load_depth_model


def _xy_to_pixels_torch(xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized [-1, 1] xy to pixel coordinates (torch, vectorized)."""
    return torch.stack(
        [(xy[:, 0] + 1.0) * (width * 0.5), (xy[:, 1] + 1.0) * (height * 0.5)], dim=-1
    )


def _project_points(
    parser: Parser, image_idx: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project COLMAP points visible in an image to pixel coordinates.

    Returns:
        pixels: (M, 2) pixel coords (torch, device)
        points_world: (M, 3) world coords for the same points (torch, device)
        depths_cam: (M,) depths in camera space (torch, device)
    """
    image_name = parser.image_names[image_idx]
    if image_name not in parser.point_indices:
        return (
            torch.empty(0, 2, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
        )

    point_indices = parser.point_indices[image_name]
    if len(point_indices) == 0:
        return (
            torch.empty(0, 2, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
        )

    points_world_np = parser.points[point_indices]

    camtoworld = torch.from_numpy(parser.camtoworlds[image_idx]).to(device=device, dtype=torch.float32)
    worldtocam = torch.linalg.inv(camtoworld)

    points_world = torch.from_numpy(points_world_np).to(device=device, dtype=torch.float32)
    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
    depths = points_cam[:, 2]
    valid_depth = depths > 0
    if not torch.any(valid_depth):
        return (
            torch.empty(0, 2, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
        )

    points_cam = points_cam[valid_depth]
    points_world = points_world[valid_depth]
    depths_cam = depths[valid_depth]

    camera_id = parser.camera_ids[image_idx]
    K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
    width, height = parser.imsize_dict[camera_id]

    pix = (K @ points_cam.T).T
    pix = pix[:, :2] / pix[:, 2:3]

    selector = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] < width)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < height)
    )
    if not torch.any(selector):
        return (
            torch.empty(0, 2, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
        )

    return pix[selector], points_world[selector], depths_cam[selector]


def lift_gaussians_to_3d(
    parser: Parser,
    gaussian_dataset: GaussianImageDataset,
    image_to_gaussians: Dict[str, GaussianFrame],
    chunk_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    mode: str = "ray",
    max_samples: Optional[int] = None,
    depth_repo: str = "depth-anything/Depth-Anything-V2-Small-hf",
    depth_cache_dir: Optional[Path] = None,
    depth_model=None,
    median_scene_depth: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map 2D gaussians to the nearest projected COLMAP points and return 3D seeds.

    Args:
        parser: COLMAP parser with intrinsics/extrinsics/points.
        gaussian_dataset: Loaded Instant-GI gaussians.
        image_to_gaussians: Mapping from image stem -> GaussianFrame.
        chunk_size: Controls memory use when finding nearest neighbors.
        device: torch device for computation (use CUDA for speed if available).
        mode: "ray" keeps the 2D direction and places the point along that ray
            at the nearest sparse COLMAP depth; "snap" would place directly on
            the nearest COLMAP point (not used here).
        max_samples: Optional cap on total lifted gaussians (applied randomly).
        depth_repo: HuggingFace identifier for Depth Anything V2.
        depth_cache_dir: Optional directory to cache dense depth maps.
        depth_model: Optional preloaded model; if None we load it once.
        median_scene_depth: Optional median depth of the scene; if provided,
            depths > 3x this value are culled.

    Returns:
        points (torch.Tensor): [N, 3] world positions for initialized gaussians.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
    """
    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []

    # Load depth model once
    depth_assets = depth_model
    if depth_assets is None:
        depth_assets = load_depth_model(repo_id=depth_repo, device=str(device))

    if (
        not isinstance(depth_assets, tuple)
        or len(depth_assets) != 2
    ):
        raise TypeError(
            "depth_model must be a (processor, model) tuple as returned by load_depth_model"
        )
    depth_processor, depth_model = depth_assets

    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if stem not in image_to_gaussians:
            continue

        pix_proj, points_world, depths_cam = _project_points(parser, idx, device=device)
        if pix_proj.numel() == 0:
            continue

        # Load gaussians for this frame.
        params = gaussian_dataset.get_gaussians_for_image(image_path)
        xy = params["xy"].to(device=device, dtype=torch.float32)
        colors = params["color"].to(device=device, dtype=torch.float32)

        camera_id = parser.camera_ids[idx]
        width, height = parser.imsize_dict[camera_id]
        K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).to(device=device, dtype=torch.float32)
        pix_gauss = _xy_to_pixels_torch(xy, width, height)

        if pix_gauss.numel() == 0:
            continue

        # Dense depth inference (with cache).
        dense_depth = infer_depth_map(
            Path(image_path),
            processor=depth_processor,
            model=depth_model,
            device=str(device),
            cache_dir=depth_cache_dir,
        )  # [1,1,H,W] on CPU (cached); move to device for sampling
        dense_depth = dense_depth.to(device=device)

        # Align dense depth to sparse COLMAP depths via RANSAC scale/shift.
        if pix_proj.numel() > 0:
            # Sample dense depth at sparse points.
            grid_sparse = pix_proj.clone()
            grid_sparse[:, 0] = (grid_sparse[:, 0] / (width - 1)) * 2 - 1
            grid_sparse[:, 1] = (grid_sparse[:, 1] / (height - 1)) * 2 - 1
            grid_sparse = grid_sparse.view(1, 1, -1, 2)
            dense_sparse = F.grid_sample(
                dense_depth, grid_sparse, align_corners=True, mode="bilinear"
            ).view(-1)

            s, t = align_depth_ransac(dense_sparse, depths_cam)
        else:
            s, t = 1.0, 0.0

        # Sample aligned dense depth at gaussian centers.
        grid_gauss = pix_gauss.clone()
        grid_gauss[:, 0] = (grid_gauss[:, 0] / (width - 1)) * 2 - 1
        grid_gauss[:, 1] = (grid_gauss[:, 1] / (height - 1)) * 2 - 1
        grid_gauss = grid_gauss.view(1, 1, -1, 2)
        dense_gauss = F.grid_sample(
            dense_depth, grid_gauss, align_corners=True, mode="bilinear"
        ).view(-1)

        depth_sel = torch.clamp(s * dense_gauss + t, min=1e-4)  # avoid zero/neg depth
        if median_scene_depth is not None:
            depth_cap = 3.0 * float(median_scene_depth)
            depth_mask = depth_sel <= depth_cap
            depth_sel = depth_sel[depth_mask]
            pix_gauss = pix_gauss[depth_mask]
            colors = colors[depth_mask]
            if depth_sel.numel() == 0:
                continue

        # camera ray directions in cam frame
        u = pix_gauss[:, 0]
        v = pix_gauss[:, 1]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        dirs_cam = torch.stack(
            [(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1
        )
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)
        pts_cam = dirs_cam * depth_sel.unsqueeze(-1)
        R = camtoworld[:3, :3]
        t = camtoworld[:3, 3]
        pts_world_sel = (R @ pts_cam.T).T + t

        all_points.append(pts_world_sel.cpu())
        all_colors.append(colors.cpu())

    if len(all_points) == 0:
        raise ValueError("gs-init found no matching gaussians to lift into 3D.")

    points_t = torch.cat(all_points, dim=0)
    colors_t = torch.cat(all_colors, dim=0)

    if max_samples is not None and len(points_t) > max_samples:
        idx = torch.randperm(len(points_t))[:max_samples]
        points_t = points_t[idx]
        colors_t = colors_t[idx]

    return points_t, colors_t


def align_depth_ransac(
    dense_depth_samples: torch.Tensor, sparse_depths: torch.Tensor
) -> Tuple[float, float]:
    """
    Fit scale/shift between dense relative depth and sparse metric depth:
        sparse ≈ s * dense + t
    """
    from sklearn.linear_model import RANSACRegressor

    x = dense_depth_samples.detach().cpu().numpy().reshape(-1, 1)
    y = sparse_depths.detach().cpu().numpy().reshape(-1, 1)
    # Filter NaNs/Infs
    mask = np.isfinite(x[:, 0]) & np.isfinite(y[:, 0])
    if len(y[mask]) > 10:
        depth_cap = np.percentile(y[mask], 95)
        mask = mask & (y[:, 0] < depth_cap)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return 1.0, 0.0

    ransac = RANSACRegressor(min_samples=2)
    ransac.fit(x, y)
    s = float(ransac.estimator_.coef_[0][0])
    t = float(ransac.estimator_.intercept_[0])
    if s <= 0:
        s = 1.0
    return s, t


def apply_fastgs_consistency_filter(
    points_world: torch.Tensor,
    source_cam_idx: int,
    parser: Parser,
    depth_assets,
    device: torch.device,
    depth_cache_dir: Optional[Path] = None,
    depth_thresh: float = 0.10,
) -> torch.Tensor:
    """
    Project lifted points into neighbor views and prune if depth is inconsistent.

    Returns:
        mask (bool tensor): True for points that are consistent in at least one neighbor.
    """
    processor, model = depth_assets
    N = len(points_world)
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    neighbor_indices = [
        i
        for i in range(source_cam_idx - 2, source_cam_idx + 3)
        if i != source_cam_idx and 0 <= i < len(parser.image_paths)
    ]
    if not neighbor_indices:
        return torch.ones(N, dtype=torch.bool, device=device)

    votes = torch.zeros(N, device=device)

    for nb_idx in neighbor_indices:
        camtoworld = torch.from_numpy(parser.camtoworlds[nb_idx]).to(device=device, dtype=torch.float32)
        worldtocam = torch.linalg.inv(camtoworld)
        K = torch.from_numpy(parser.Ks_dict[parser.camera_ids[nb_idx]]).to(device=device, dtype=torch.float32)
        width, height = parser.imsize_dict[parser.camera_ids[nb_idx]]

        pts_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
        depths = pts_cam[:, 2]
        valid = depths > 0
        if not torch.any(valid):
            continue

        pts_cam = pts_cam[valid]
        depths = depths[valid]

        pix = (K @ pts_cam.T).T
        pix = pix[:, :2] / pix[:, 2:3]

        in_frame = (
            (pix[:, 0] >= 0)
            & (pix[:, 0] < width)
            & (pix[:, 1] >= 0)
            & (pix[:, 1] < height)
        )
        if not torch.any(in_frame):
            continue

        pix = pix[in_frame]
        depths = depths[in_frame]
        idx_valid = valid.nonzero(as_tuple=False).flatten()[in_frame]

        # Load neighbor dense depth
        dense_depth = infer_depth_map(
            Path(parser.image_paths[nb_idx]),
            processor=processor,
            model=model,
            device=str(device),
            cache_dir=depth_cache_dir,
        ).to(device=device)

        grid = pix.clone()
        grid[:, 0] = (grid[:, 0] / (width - 1)) * 2 - 1
        grid[:, 1] = (grid[:, 1] / (height - 1)) * 2 - 1
        grid = grid.view(1, 1, -1, 2)
        sampled = F.grid_sample(dense_depth, grid, align_corners=True, mode="bilinear").view(-1)

        denom = sampled.abs() + 1e-6
        rel_err = (depths - sampled).abs() / denom
        is_consistent = rel_err < depth_thresh

        votes[idx_valid[is_consistent]] += 1

    return votes >= 1
