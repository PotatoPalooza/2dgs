"""
Lift 2D Instant-GI gaussians into 3D initialization points.

We project COLMAP sparse points into each training image, then for every 2D
gaussian center we pick the nearest projected COLMAP point in pixel space and
use that 3D position (plus the gaussian RGB) to seed a 3D gaussian. This allows
an initialization that mirrors the 2D gaussian coverage instead of sampling the
COLMAP cloud directly.

If you have dense depth maps aligned to COLMAP, you can instead use init_type
``gs-depth`` to backproject 2D gaussians using per-pixel depths.

The heavy lifting (distance search) is done with torch and chunked so we can
handle ~10M gaussians without blowing memory. Pass a CUDA device to run the
nearest-neighbor search on GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

from datasets.colmap import Parser
from gaussian_image_dataset import GaussianFrame, GaussianImageDataset
from depth_loader import DepthLoader


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
                # DA3 (and COLMAP) depth is camera-space Z (planar depth).
                # Backproject using z * K^{-1}[u,v,1]^T (no ray normalization).
                dirs_cam = torch.stack(
                    [(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1
                )
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


def _sample_depth_bilinear(
    depth_map: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Bilinearly sample a depth map at floating pixel coordinates.

    Args:
        depth_map: (H, W) float depth tensor on device.
        u, v: (B,) pixel x/y coordinates.

    Returns:
        (B,) sampled depths.
    """
    h, w = depth_map.shape
    u0 = torch.floor(u).long().clamp(0, w - 1)
    v0 = torch.floor(v).long().clamp(0, h - 1)
    u1 = (u0 + 1).clamp(0, w - 1)
    v1 = (v0 + 1).clamp(0, h - 1)

    du = (u - u0.float()).clamp(0.0, 1.0)
    dv = (v - v0.float()).clamp(0.0, 1.0)

    d00 = depth_map[v0, u0]
    d01 = depth_map[v0, u1]
    d10 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]

    d0 = d00 * (1.0 - du) + d01 * du
    d1 = d10 * (1.0 - du) + d11 * du
    return d0 * (1.0 - dv) + d1 * dv


def lift_gaussians_to_3d_with_dense_depth(
    parser: Parser,
    gaussian_dataset: GaussianImageDataset,
    image_to_gaussians: Dict[str, GaussianFrame],
    depth_loader: DepthLoader,
    chunk_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    min_depth: float = 1e-3,
    depth_percentile: float = 95.0,
    max_depth_scale: float = 5.0,
    max_depth_abs: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backproject 2D gaussians into 3D using dense depth maps.

    For each training image, we load its aligned depth map and sample depth at
    each 2D gaussian center, then backproject along camera rays into world space.

    Args:
        parser: COLMAP parser with intrinsics/extrinsics.
        gaussian_dataset: Loaded Instant-GI gaussians.
        image_to_gaussians: Mapping from image stem -> GaussianFrame.
        depth_loader: DepthLoader pointing at per-image depth maps.
        chunk_size: Controls memory use when processing gaussians per frame.
        device: torch device for computation (use CUDA for speed if available).
        min_depth: Minimum valid depth in meters.
        max_samples: Optional cap on total lifted gaussians (applied randomly).

    Returns:
        points (torch.Tensor): [N, 3] world positions for initialized gaussians.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
    """
    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []
    all_scales_log: List[torch.Tensor] = []
    all_opacities_logit: List[torch.Tensor] = []

    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if stem not in image_to_gaussians:
            continue

        camera_id = parser.camera_ids[idx]
        width, height = parser.imsize_dict[camera_id]

        depth_np = depth_loader.load_depth(stem, target_hw=(height, width))
        if depth_np is None:
            continue

        # If parser normalized world space, apply the same similarity scale to depth.
        # Normalization scales all translations/points by s, so planar depths must be scaled by s too.
        depth_scale = 1.0
        if getattr(parser, "normalize", False):
            transform = getattr(parser, "transform", None)
            if transform is not None:
                try:
                    depth_scale = float(np.linalg.norm(np.asarray(transform)[0, :3]))
                except Exception:
                    depth_scale = 1.0
        if depth_scale != 1.0:
            depth_np = depth_np * depth_scale

        # Filter unbounded/sky regions: DA3 sets sky to very large depth.
        valid_np = depth_np[np.isfinite(depth_np) & (depth_np > min_depth)]
        if valid_np.size == 0:
            continue
        p_clip = float(np.percentile(valid_np, depth_percentile))
        scene_scale = float(getattr(parser, "scene_scale", 0.0) or 0.0)
        scene_clip = scene_scale * max_depth_scale if scene_scale > 0 else p_clip
        max_depth = min(p_clip, scene_clip) if scene_clip > 0 else p_clip
        if max_depth_abs is not None:
            max_depth = min(max_depth, float(max_depth_abs))
        max_depth = max(max_depth, min_depth)

        depth_map = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)

        params = gaussian_dataset.get_gaussians_for_image(image_path)
        xy = params["xy"].to(device=device, dtype=torch.float32)
        colors = params["color"].to(device=device, dtype=torch.float32)
        scales_2d = params["scale"].to(device=device, dtype=torch.float32)
        opacity_2d = params["opacity"].to(device=device, dtype=torch.float32)

        K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).to(device=device, dtype=torch.float32)
        pix_gauss = _xy_to_pixels_torch(xy, width, height)

        if pix_gauss.numel() == 0:
            continue

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = camtoworld[:3, :3]
        t = camtoworld[:3, 3]

        for start in range(0, len(pix_gauss), chunk_size):
            end = min(start + chunk_size, len(pix_gauss))
            g_chunk = pix_gauss[start:end]
            c_chunk = colors[start:end]
            s2d_chunk = scales_2d[start:end]
            o2d_chunk = opacity_2d[start:end]

            u = g_chunk[:, 0]
            v = g_chunk[:, 1]
            depth_sel = _sample_depth_bilinear(depth_map, u, v)
            valid = (depth_sel > min_depth) & (depth_sel <= max_depth)
            if not torch.any(valid):
                continue

            u = u[valid]
            v = v[valid]
            depth_sel = depth_sel[valid]
            c_chunk = c_chunk[valid]
            s2d_chunk = s2d_chunk[valid]
            o2d_chunk = o2d_chunk[valid]

            # DA3 depth maps are planar camera-Z depths.
            dirs_cam = torch.stack(
                [(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1
            )
            pts_cam = dirs_cam * depth_sel.unsqueeze(-1)
            pts_world_sel = (R @ pts_cam.T).T + t

            # Map 2D gaussian scale (in pixels) to a 3D isotropic stddev.
            # Approx: sigma_world_x ~ z * sigma_px_x / fx; sigma_world_y ~ z * sigma_px_y / fy
            sigma_px_x = s2d_chunk[:, 0].clamp(min=0.5)
            sigma_px_y = s2d_chunk[:, 1].clamp(min=0.5)
            sigma_world_x = depth_sel * (sigma_px_x / fx)
            sigma_world_y = depth_sel * (sigma_px_y / fy)
            sigma_world = torch.sqrt(torch.clamp(sigma_world_x * sigma_world_y, min=1e-12))
            sigma_world = sigma_world.clamp(min=1e-6)
            scales_log = torch.log(sigma_world).unsqueeze(-1).repeat(1, 3)

            # Map 2D opacity (assumed in [0,1]) to logit parameterization used by gsplat.
            o = o2d_chunk.squeeze(-1)
            o = torch.clamp(o, 1e-4, 1.0 - 1e-4)
            opacities_logit = torch.logit(o)

            all_points.append(pts_world_sel.cpu())
            all_colors.append(c_chunk.cpu())
            all_scales_log.append(scales_log.cpu())
            all_opacities_logit.append(opacities_logit.cpu())

    if len(all_points) == 0:
        raise ValueError("gs-depth found no matching gaussians with valid dense depths.")

    points_t = torch.cat(all_points, dim=0)
    colors_t = torch.cat(all_colors, dim=0)
    scales_t = torch.cat(all_scales_log, dim=0)
    opacities_t = torch.cat(all_opacities_logit, dim=0)

    if max_samples is not None and len(points_t) > max_samples:
        idx_sel = torch.randperm(len(points_t))[:max_samples]
        points_t = points_t[idx_sel]
        colors_t = colors_t[idx_sel]
        scales_t = scales_t[idx_sel]
        opacities_t = opacities_t[idx_sel]

    return points_t, colors_t, scales_t, opacities_t


def lift_random_points_with_dense_depth(
    parser: Parser,
    depth_loader: DepthLoader,
    num_points: int,
    device: torch.device = torch.device("cpu"),
    chunk_size: int = 8192,
    min_depth: float = 1e-3,
    depth_percentile: float = 95.0,
    max_depth_scale: float = 5.0,
    max_depth_abs: Optional[float] = None,
    max_attempts: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample image pixels and backproject with dense depth.

    This is similar to gs-depth, but does not require 2D gaussians. It samples
    up to ``num_points`` random pixels across all views that have depth maps.

    Returns:
        points (torch.Tensor): [N, 3] world positions.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
    """
    valid_views: List[int] = []
    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if depth_loader.depth_path_for_stem(stem) is not None:
            valid_views.append(idx)

    if not valid_views:
        raise ValueError("rand-depth found no views with dense depth maps.")

    per_view_target = int(np.ceil(num_points / len(valid_views)))

    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []

    for idx in valid_views:
        image_path = parser.image_paths[idx]
        stem = Path(image_path).stem

        camera_id = parser.camera_ids[idx]
        width, height = parser.imsize_dict[camera_id]

        depth_np = depth_loader.load_depth(stem, target_hw=(height, width))
        if depth_np is None:
            continue

        # Apply normalization scale if needed (same as gs-depth).
        depth_scale = 1.0
        if getattr(parser, "normalize", False):
            transform = getattr(parser, "transform", None)
            if transform is not None:
                try:
                    depth_scale = float(np.linalg.norm(np.asarray(transform)[0, :3]))
                except Exception:
                    depth_scale = 1.0
        if depth_scale != 1.0:
            depth_np = depth_np * depth_scale

        valid_np = depth_np[np.isfinite(depth_np) & (depth_np > min_depth)]
        if valid_np.size == 0:
            continue
        p_clip = float(np.percentile(valid_np, depth_percentile))
        scene_scale = float(getattr(parser, "scene_scale", 0.0) or 0.0)
        scene_clip = scene_scale * max_depth_scale if scene_scale > 0 else p_clip
        max_depth = min(p_clip, scene_clip) if scene_clip > 0 else p_clip
        if max_depth_abs is not None:
            max_depth = min(max_depth, float(max_depth_abs))
        max_depth = max(max_depth, min_depth)

        depth_map = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
        K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).to(device=device, dtype=torch.float32)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = camtoworld[:3, :3]
        t = camtoworld[:3, 3]

        # Load image for color sampling.
        try:
            img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            img_np = None

        collected_pts: List[torch.Tensor] = []
        collected_cols: List[torch.Tensor] = []

        remaining = per_view_target
        for _ in range(max_attempts):
            if remaining <= 0:
                break
            sample_n = max(chunk_size, remaining * 2)
            u = torch.rand(sample_n, device=device) * (width - 1)
            v = torch.rand(sample_n, device=device) * (height - 1)

            depth_sel = _sample_depth_bilinear(depth_map, u, v)
            valid = (depth_sel > min_depth) & (depth_sel <= max_depth)
            if not torch.any(valid):
                continue

            u_v = u[valid]
            v_v = v[valid]
            d_v = depth_sel[valid]

            dirs_cam = torch.stack(
                [(u_v - cx) / fx, (v_v - cy) / fy, torch.ones_like(u_v)], dim=-1
            )
            pts_cam = dirs_cam * d_v.unsqueeze(-1)
            pts_world = (R @ pts_cam.T).T + t

            take = min(remaining, pts_world.shape[0])
            if take <= 0:
                break
            collected_pts.append(pts_world[:take].cpu())

            if img_np is not None:
                uu = u_v[:take].round().long().clamp(0, width - 1).cpu().numpy()
                vv = v_v[:take].round().long().clamp(0, height - 1).cpu().numpy()
                cols = img_np[vv, uu].astype(np.float32) / 255.0
                collected_cols.append(torch.from_numpy(cols))
            else:
                collected_cols.append(torch.zeros((take, 3), dtype=torch.float32))

            remaining -= take

        if collected_pts:
            all_points.append(torch.cat(collected_pts, dim=0))
            all_colors.append(torch.cat(collected_cols, dim=0))

    if not all_points:
        raise ValueError("rand-depth found no valid samples after filtering.")

    points_t = torch.cat(all_points, dim=0)
    colors_t = torch.cat(all_colors, dim=0)

    # Cap to requested num_points.
    if points_t.shape[0] > num_points:
        idx_sel = torch.randperm(points_t.shape[0])[:num_points]
        points_t = points_t[idx_sel]
        colors_t = colors_t[idx_sel]

    return points_t, colors_t


def lift_prob_points_with_dense_depth(
    parser: Parser,
    gaussian_dataset: GaussianImageDataset,
    depth_loader: DepthLoader,
    num_points: int,
    device: torch.device = torch.device("cpu"),
    chunk_size: int = 8192,
    min_depth: float = 1e-3,
    depth_percentile: float = 95.0,
    max_depth_scale: float = 5.0,
    max_depth_abs: Optional[float] = None,
    max_attempts: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample pixels per view according to a probability field (gt_pf.npy) and
    backproject into world space using dense depth maps.

    The probability field is loaded from the same directory as the Instant-GI
    gaussian checkpoint for that frame:
        <frame>/<init_method>/gt_pf.npy

    This behaves similarly to rand-depth, but samples x/y using the probability
    field instead of uniform random sampling.

    Returns:
        points (torch.Tensor): [N, 3] world positions.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
    """
    valid_views: List[int] = []
    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if depth_loader.depth_path_for_stem(stem) is None:
            continue
        frame = gaussian_dataset.get_frame(stem)
        if frame is None:
            continue
        pf_path = frame.path.parent / "gt_pf.npy"
        if pf_path.exists():
            valid_views.append(idx)

    if not valid_views:
        raise ValueError("gs-prob found no views with both depth maps and gt_pf.npy.")

    per_view_target = int(np.ceil(num_points / len(valid_views)))

    all_points: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []

    for idx in valid_views:
        image_path = parser.image_paths[idx]
        stem = Path(image_path).stem

        camera_id = parser.camera_ids[idx]
        width, height = parser.imsize_dict[camera_id]

        depth_np = depth_loader.load_depth(stem, target_hw=(height, width))
        if depth_np is None:
            continue

        # Apply normalization scale if needed (same as gs-depth).
        depth_scale = 1.0
        if getattr(parser, "normalize", False):
            transform = getattr(parser, "transform", None)
            if transform is not None:
                try:
                    depth_scale = float(np.linalg.norm(np.asarray(transform)[0, :3]))
                except Exception:
                    depth_scale = 1.0
        if depth_scale != 1.0:
            depth_np = depth_np * depth_scale

        valid_np = depth_np[np.isfinite(depth_np) & (depth_np > min_depth)]
        if valid_np.size == 0:
            continue
        p_clip = float(np.percentile(valid_np, depth_percentile))
        scene_scale = float(getattr(parser, "scene_scale", 0.0) or 0.0)
        scene_clip = scene_scale * max_depth_scale if scene_scale > 0 else p_clip
        max_depth = min(p_clip, scene_clip) if scene_clip > 0 else p_clip
        if max_depth_abs is not None:
            max_depth = min(max_depth, float(max_depth_abs))
        max_depth = max(max_depth, min_depth)

        frame = gaussian_dataset.get_frame(stem)
        if frame is None:
            continue
        pf_path = frame.path.parent / "gt_pf.npy"
        if not pf_path.exists():
            continue
        pf = np.load(str(pf_path)).astype(np.float32)
        if pf.ndim == 3 and pf.shape[0] == 1:
            pf = pf[0]
        if pf.shape != (height, width):
            pf_img = Image.fromarray(pf)
            pf_img = pf_img.resize((width, height), resample=Image.BILINEAR)
            pf = np.array(pf_img, dtype=np.float32)
        pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
        pf = np.clip(pf, 0.0, None)
        pf_flat = torch.from_numpy(pf.reshape(-1))
        if float(pf_flat.sum()) <= 0.0:
            pf_flat = torch.ones_like(pf_flat)

        depth_map = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
        K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).to(device=device, dtype=torch.float32)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = camtoworld[:3, :3]
        t = camtoworld[:3, 3]

        try:
            img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            img_np = None

        collected_pts: List[torch.Tensor] = []
        collected_cols: List[torch.Tensor] = []

        remaining = per_view_target
        for _ in range(max_attempts):
            if remaining <= 0:
                break
            sample_n = max(chunk_size, remaining * 2)

            idx_flat = torch.multinomial(pf_flat, sample_n, replacement=True)
            u_idx = (idx_flat % width).to(device=device, dtype=torch.long)
            v_idx = (idx_flat // width).to(device=device, dtype=torch.long)

            depth_sel = depth_map[v_idx, u_idx]
            valid = (depth_sel > min_depth) & (depth_sel <= max_depth) & torch.isfinite(depth_sel)
            if not torch.any(valid):
                continue

            u_v = u_idx[valid].to(dtype=torch.float32)
            v_v = v_idx[valid].to(dtype=torch.float32)
            d_v = depth_sel[valid]

            dirs_cam = torch.stack(
                [(u_v - cx) / fx, (v_v - cy) / fy, torch.ones_like(u_v)], dim=-1
            )
            pts_cam = dirs_cam * d_v.unsqueeze(-1)
            pts_world = (R @ pts_cam.T).T + t

            take = min(remaining, pts_world.shape[0])
            if take <= 0:
                break
            collected_pts.append(pts_world[:take].cpu())

            if img_np is not None:
                uu = u_idx[valid][:take].clamp(0, width - 1).cpu().numpy()
                vv = v_idx[valid][:take].clamp(0, height - 1).cpu().numpy()
                cols = img_np[vv, uu].astype(np.float32) / 255.0
                collected_cols.append(torch.from_numpy(cols))
            else:
                collected_cols.append(torch.zeros((take, 3), dtype=torch.float32))

            remaining -= take

        if collected_pts:
            all_points.append(torch.cat(collected_pts, dim=0))
            all_colors.append(torch.cat(collected_cols, dim=0))

    if not all_points:
        raise ValueError("gs-prob found no valid samples after filtering.")

    points_t = torch.cat(all_points, dim=0)
    colors_t = torch.cat(all_colors, dim=0)

    if points_t.shape[0] > num_points:
        idx_sel = torch.randperm(points_t.shape[0])[:num_points]
        points_t = points_t[idx_sel]
        colors_t = colors_t[idx_sel]

    return points_t, colors_t


def lift_prob_multi_points_with_dense_depth(
    parser: Parser,
    gaussian_dataset: GaussianImageDataset,
    depth_loader: DepthLoader,
    num_points: int,
    device: torch.device = torch.device("cpu"),
    oversample: float = 4.0,
    voxel_size: Optional[float] = None,
    min_views: int = 2,
    view_power: float = 0.5,
    jitter_frac: float = 0.0,
    seed: int = 0,
    chunk_size: int = 8192,
    min_depth: float = 1e-3,
    depth_percentile: float = 95.0,
    max_depth_scale: float = 5.0,
    max_depth_abs: Optional[float] = None,
    max_attempts: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-view consistent sampling from per-view probability fields (gt_pf.npy).

    This initializer backprojects candidate points from each view using aligned
    dense depth maps, quantizes into world-space voxels, and prefers voxels that
    receive high probability support from many views.

    Returns:
        points (torch.Tensor): [N, 3] world positions.
        colors (torch.Tensor): [N, 3] RGB in [0, 1].
        scales_log (torch.Tensor): [N, 3] log stddev used to avoid KNN init.
    """

    def _pack_voxel_keys(vox: torch.Tensor) -> torch.Tensor:
        """
        Pack integer voxel coordinates into a 64-bit key.

        Uses 21 bits per axis (signed range [-2^20, 2^20-1]). If any coordinate
        is out of range, falls back to a hashed key (rare with reasonable voxel_size).
        """
        if vox.numel() == 0:
            return torch.empty((0,), device=vox.device, dtype=torch.int64)
        offset = 1 << 20
        max_abs = offset - 1
        if torch.any(torch.abs(vox) > max_abs):
            x = vox[:, 0].to(torch.int64)
            y = vox[:, 1].to(torch.int64)
            z = vox[:, 2].to(torch.int64)
            return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        vx = (vox[:, 0] + offset).to(torch.int64)
        vy = (vox[:, 1] + offset).to(torch.int64)
        vz = (vox[:, 2] + offset).to(torch.int64)
        return vx | (vy << 21) | (vz << 42)

    if oversample <= 0:
        raise ValueError(f"oversample must be > 0, got {oversample}")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")
    if min_views <= 0:
        raise ValueError(f"min_views must be > 0, got {min_views}")

    scene_scale = float(getattr(parser, "scene_scale", 0.0) or 0.0)
    if voxel_size is None:
        voxel_size = scene_scale / 256.0 if scene_scale > 0 else 0.01
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    valid_views: List[int] = []
    for idx, image_path in enumerate(parser.image_paths):
        stem = Path(image_path).stem
        if depth_loader.depth_path_for_stem(stem) is None:
            continue
        frame = gaussian_dataset.get_frame(stem)
        if frame is None:
            continue
        pf_path = frame.path.parent / "gt_pf.npy"
        if pf_path.exists():
            valid_views.append(idx)

    if not valid_views:
        raise ValueError("gs-prob-multi found no views with both depth maps and gt_pf.npy.")

    per_view_target = int(np.ceil(num_points / len(valid_views)))
    per_view_samples = int(np.ceil(per_view_target * float(oversample)))
    per_view_samples = max(1, per_view_samples)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    all_keys: List[torch.Tensor] = []
    all_centroids: List[torch.Tensor] = []
    all_colors: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []

    for idx in valid_views:
        image_path = parser.image_paths[idx]
        stem = Path(image_path).stem

        camera_id = parser.camera_ids[idx]
        width, height = parser.imsize_dict[camera_id]

        depth_np = depth_loader.load_depth(stem, target_hw=(height, width))
        if depth_np is None:
            continue

        # Apply normalization scale if needed (same as gs-depth).
        depth_scale = 1.0
        if getattr(parser, "normalize", False):
            transform = getattr(parser, "transform", None)
            if transform is not None:
                try:
                    depth_scale = float(np.linalg.norm(np.asarray(transform)[0, :3]))
                except Exception:
                    depth_scale = 1.0
        if depth_scale != 1.0:
            depth_np = depth_np * depth_scale

        valid_np = depth_np[np.isfinite(depth_np) & (depth_np > min_depth)]
        if valid_np.size == 0:
            continue
        p_clip = float(np.percentile(valid_np, depth_percentile))
        scene_clip = scene_scale * max_depth_scale if scene_scale > 0 else p_clip
        max_depth = min(p_clip, scene_clip) if scene_clip > 0 else p_clip
        if max_depth_abs is not None:
            max_depth = min(max_depth, float(max_depth_abs))
        max_depth = max(max_depth, min_depth)

        frame = gaussian_dataset.get_frame(stem)
        if frame is None:
            continue
        pf_path = frame.path.parent / "gt_pf.npy"
        if not pf_path.exists():
            continue
        pf = np.load(str(pf_path)).astype(np.float32)
        if pf.ndim == 3 and pf.shape[0] == 1:
            pf = pf[0]
        if pf.shape != (height, width):
            pf_img = Image.fromarray(pf)
            pf_img = pf_img.resize((width, height), resample=Image.BILINEAR)
            pf = np.array(pf_img, dtype=np.float32)
        pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
        pf = np.clip(pf, 0.0, None)
        pf_flat = torch.from_numpy(pf.reshape(-1)).to(dtype=torch.float32, device="cpu")
        if float(pf_flat.sum()) <= 0.0:
            pf_flat = torch.ones_like(pf_flat)
        pf_max = float(torch.max(pf_flat).item())
        if pf_max <= 0.0:
            pf_max = 1.0

        depth_map = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
        K = torch.from_numpy(parser.Ks_dict[camera_id]).to(device=device, dtype=torch.float32)
        camtoworld = torch.from_numpy(parser.camtoworlds[idx]).to(device=device, dtype=torch.float32)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R = camtoworld[:3, :3]
        t = camtoworld[:3, 3]

        try:
            img_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            img_np = None

        collected_pts: List[torch.Tensor] = []
        collected_cols: List[torch.Tensor] = []
        collected_w: List[torch.Tensor] = []

        remaining = per_view_samples
        for _ in range(max_attempts):
            if remaining <= 0:
                break
            sample_n = max(chunk_size, remaining * 2)

            idx_flat = torch.multinomial(pf_flat, sample_n, replacement=True, generator=gen)
            u_cpu = (idx_flat % width).to(dtype=torch.long, device="cpu")
            v_cpu = (idx_flat // width).to(dtype=torch.long, device="cpu")
            w_cpu = (pf_flat[idx_flat] / pf_max).to(dtype=torch.float32, device="cpu")

            u_idx = u_cpu.to(device=device)
            v_idx = v_cpu.to(device=device)
            w = w_cpu.to(device=device)

            depth_sel = depth_map[v_idx, u_idx]
            valid = (depth_sel > min_depth) & (depth_sel <= max_depth) & torch.isfinite(depth_sel)
            if not torch.any(valid):
                continue

            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            take = min(remaining, int(valid_idx.shape[0]))
            if take <= 0:
                continue
            valid_idx = valid_idx[:take]
            valid_idx_cpu = valid_idx.cpu()

            u_v = u_idx[valid_idx].to(dtype=torch.float32)
            v_v = v_idx[valid_idx].to(dtype=torch.float32)
            d_v = depth_sel[valid_idx]
            w_v = w[valid_idx]

            dirs_cam = torch.stack(
                [(u_v - cx) / fx, (v_v - cy) / fy, torch.ones_like(u_v)], dim=-1
            )
            pts_cam = dirs_cam * d_v.unsqueeze(-1)
            pts_world = (R @ pts_cam.T).T + t

            if img_np is not None:
                uu = u_cpu[valid_idx_cpu].clamp(0, width - 1).numpy()
                vv = v_cpu[valid_idx_cpu].clamp(0, height - 1).numpy()
                cols = img_np[vv, uu].astype(np.float32) / 255.0
                cols_t = torch.from_numpy(cols).to(device=device, dtype=torch.float32)
            else:
                cols_t = torch.zeros((take, 3), device=device, dtype=torch.float32)

            collected_pts.append(pts_world)
            collected_cols.append(cols_t)
            collected_w.append(w_v)
            remaining -= take

        if not collected_pts:
            continue

        pts_all = torch.cat(collected_pts, dim=0)
        cols_all = torch.cat(collected_cols, dim=0)
        w_all = torch.cat(collected_w, dim=0).clamp(min=0.0)

        vox = torch.floor(pts_all / float(voxel_size)).to(torch.int64)
        keys = _pack_voxel_keys(vox)

        unique_keys, inv = torch.unique(keys, return_inverse=True)
        n_vox = int(unique_keys.shape[0])

        ones = torch.ones_like(inv, dtype=torch.float32, device=device)
        hits = torch.zeros(n_vox, device=device, dtype=torch.float32).index_add_(0, inv, ones)
        w_sum = torch.zeros(n_vox, device=device, dtype=torch.float32).index_add_(0, inv, w_all)

        p_sum = torch.zeros((n_vox, 3), device=device, dtype=torch.float32).index_add_(
            0, inv, pts_all * w_all.unsqueeze(-1)
        )
        c_sum = torch.zeros((n_vox, 3), device=device, dtype=torch.float32).index_add_(
            0, inv, cols_all * w_all.unsqueeze(-1)
        )

        denom = w_sum.clamp(min=1e-8).unsqueeze(-1)
        centroids = p_sum / denom
        colors = c_sum / denom
        w_mean = w_sum / hits.clamp(min=1.0)

        all_keys.append(unique_keys.cpu())
        all_centroids.append(centroids.cpu())
        all_colors.append(colors.cpu().clamp(0.0, 1.0))
        all_weights.append(w_mean.cpu())

    if not all_keys:
        raise ValueError("gs-prob-multi found no valid samples after filtering.")

    keys_all = torch.cat(all_keys, dim=0)
    centroids_all = torch.cat(all_centroids, dim=0)
    colors_all = torch.cat(all_colors, dim=0)
    w_all = torch.cat(all_weights, dim=0).clamp(min=0.0)

    global_keys, inv_global = torch.unique(keys_all, return_inverse=True)
    n_global = int(global_keys.shape[0])

    one = torch.ones_like(inv_global, dtype=torch.float32)
    view_count = torch.zeros(n_global, dtype=torch.float32).index_add_(0, inv_global, one)
    w_sum = torch.zeros(n_global, dtype=torch.float32).index_add_(0, inv_global, w_all)

    centroid_sum = torch.zeros((n_global, 3), dtype=torch.float32).index_add_(
        0, inv_global, centroids_all * w_all.unsqueeze(-1)
    )
    color_sum = torch.zeros((n_global, 3), dtype=torch.float32).index_add_(
        0, inv_global, colors_all * w_all.unsqueeze(-1)
    )

    denom = w_sum.clamp(min=1e-8).unsqueeze(-1)
    centroids_g = centroid_sum / denom
    colors_g = color_sum / denom

    keep = view_count >= float(min_views)
    if not torch.any(keep):
        raise ValueError(
            f"gs-prob-multi found no voxels with min_views={min_views}. "
            "Try lowering min_views or increasing oversample/voxel_size."
        )

    cand_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    scores = w_sum[keep] * torch.pow(view_count[keep], float(view_power))
    if float(scores.sum()) <= 0.0:
        scores = torch.ones_like(scores)

    replace = int(cand_idx.shape[0]) < int(num_points)
    sel = torch.multinomial(scores, int(num_points), replacement=replace, generator=gen)
    chosen = cand_idx[sel]

    points_t = centroids_g[chosen]
    colors_t = colors_g[chosen].clamp(0.0, 1.0)

    if jitter_frac > 0.0:
        jitter = (torch.rand(points_t.shape, generator=gen) - 0.5) * (float(voxel_size) * float(jitter_frac))
        points_t = points_t + jitter.to(points_t.dtype)

    base_sigma = max(float(voxel_size) * 0.5, 1e-6)
    scales_log = torch.full(
        (points_t.shape[0], 3), float(np.log(base_sigma)), dtype=torch.float32
    )

    return points_t, colors_t, scales_log
