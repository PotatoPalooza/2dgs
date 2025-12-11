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

import numpy as np
import torch

from datasets.colmap import Parser
from gaussian_image_dataset import GaussianFrame, GaussianImageDataset


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

    Returns:
        points (torch.Tensor): [N, 3] world positions for initialized gaussians.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
    """
    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []

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

        # Find nearest projected COLMAP point for every gaussian center (chunked).
        for start in range(0, len(pix_gauss), chunk_size):
            end = min(start + chunk_size, len(pix_gauss))
            g_chunk = pix_gauss[start:end]  # (B, 2)
            # torch.cdist uses broadcasting + optimized kernel; squared distances are enough for argmin.
            dist = torch.cdist(g_chunk, pix_proj, p=2)  # (B, M)
            nearest_idx = torch.argmin(dist, dim=1)

            if mode == "snap":
                pts_world_sel = points_world[nearest_idx]
            else:  # "ray": keep direction, use depth prior
                depth_sel = depths_cam[nearest_idx]  # (B,)
                # camera ray directions in cam frame
                u = g_chunk[:, 0]
                v = g_chunk[:, 1]
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
            all_colors.append(colors[start:end].cpu())

    if len(all_points) == 0:
        raise ValueError("gs-init found no matching gaussians to lift into 3D.")

    points_t = torch.cat(all_points, dim=0)
    colors_t = torch.cat(all_colors, dim=0)

    if max_samples is not None and len(points_t) > max_samples:
        idx = torch.randperm(len(points_t))[:max_samples]
        points_t = points_t[idx]
        colors_t = colors_t[idx]

    return points_t, colors_t
