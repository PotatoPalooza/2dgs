"""
Lift 2D Instant-GI gaussians into 3D using globally aligned Dust3r depths.

We use Dust3r (via mini-dust3r) to produce multi-view-consistent dense depth
maps, sample those depths at the 2D gaussian centers, and unproject along the
camera rays to obtain 3D seeds. No per-view RANSAC or consistency voting is
needed because Dust3r already aligns depths across the set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from datasets.colmap import Parser
from gaussian_image_dataset import GaussianFrame, GaussianImageDataset
from dust3r_depth import compute_aligned_depths, load_dust3r_model


def _xy_to_pixels_torch(xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized [-1, 1] xy to pixel coordinates (torch, vectorized)."""
    return torch.stack(
        [(xy[:, 0] + 1.0) * (width * 0.5), (xy[:, 1] + 1.0) * (height * 0.5)], dim=-1
    )


def lift_gaussians_to_3d(
    parser: Parser,
    gaussian_dataset: GaussianImageDataset,
    image_to_gaussians: Dict[str, GaussianFrame],
    chunk_size: int = 4096,  # kept for API compatibility; unused
    device: torch.device = torch.device("cpu"),
    mode: str = "ray",  # kept for API compatibility; unused
    max_samples: Optional[int] = None,
    depth_repo: str = "dust3r/mini-dust3r",
    depth_cache_dir: Optional[Path] = None,
    depth_model=None,
    median_scene_depth: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lift 2D gaussians using Dust3r dense depths.
    """
    if not image_to_gaussians:
        raise ValueError("No images/gaussians provided for gs-init lifting.")

    # Dust3r dense depths (aligned across views)
    model = depth_model or load_dust3r_model(model_id=depth_repo, device=str(device))
    depth_maps = compute_aligned_depths(
        parser, model, device=device, cache_dir=depth_cache_dir
    )

    # Estimate median depth if not provided
    if median_scene_depth is None and depth_maps:
        vals = []
        for d in depth_maps.values():
            valid = d[d > 0]
            if valid.numel() > 0:
                vals.append(valid.view(-1))
        if vals:
            median_scene_depth = torch.cat(vals).median().item()

    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []

    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if stem not in image_to_gaussians:
            continue

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

        dense_depth = depth_maps.get(image_path)
        if dense_depth is None:
            continue
        dense_depth = dense_depth.to(device=device)

        grid_gauss = pix_gauss.clone()
        grid_gauss[:, 0] = (grid_gauss[:, 0] / (width - 1)) * 2 - 1
        grid_gauss[:, 1] = (grid_gauss[:, 1] / (height - 1)) * 2 - 1
        grid_gauss = grid_gauss.view(1, 1, -1, 2)
        depth_sel = F.grid_sample(
            dense_depth, grid_gauss, align_corners=True, mode="bilinear"
        ).view(-1)
        depth_sel = torch.clamp(depth_sel, min=1e-4)

        if median_scene_depth is not None:
            depth_cap = 3.0 * float(median_scene_depth)
            mask = depth_sel <= depth_cap
            depth_sel = depth_sel[mask]
            pix_gauss = pix_gauss[mask]
            colors = colors[mask]
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
