import math
import os
import numpy as np
import torch
from scipy.spatial import Delaunay
from generalizable_model.utils import add_boundary_points,add_boundary_points_torch, min_bounding_ellipse, dither_image, neighbors_process
from ellipse_fit import fit_ellipses
import cupy as cp
from cupyx.scipy.spatial import Delaunay as DelaunayGPU
import time
import math


# pool = None
# def init_pool(cpu_num):
#     global pool
#     pool = torch.multiprocessing.Pool(cpu_num)

def ellipse_filter(ellipses_size):
    # ellipse_size: [N, 2]
    less_mask = ellipses_size < 1
    less_mask = less_mask.any(dim=1)
    # great mask
    mean = ellipses_size.mean()
    std = ellipses_size.std()
    great_mask = ellipses_size > (mean + std)
    great_mask = great_mask.any(dim=1)
    mask = less_mask | great_mask
    return mask
    
    
def simplices_to_neighbors(simplices):
    """
    Compute the neighbors for each triangle in the triangulation using torch (CUDA).
    Returns an array of shape (n_tri, 3), where each entry is the index of the neighboring triangle
    sharing the corresponding edge, or the triangle's own index if there is no neighbor.
    """
    device = simplices.device
    n_tri = simplices.shape[0]
    # 1. Construct three edges for each triangle (n_tri, 3, 2)
    edges = torch.stack([
        torch.stack([simplices[:, 0], simplices[:, 1]], dim=1),
        torch.stack([simplices[:, 1], simplices[:, 2]], dim=1),
        torch.stack([simplices[:, 2], simplices[:, 0]], dim=1)
    ], dim=1)
    # 2. Sort to ensure undirected edge uniqueness
    edges, _ = torch.sort(edges, dim=2)
    # 3. Flatten to (n_tri*3, 2)
    flat_edges = edges.reshape(-1, 2)
    tri_ids = torch.arange(n_tri, device=device).repeat_interleave(3)
    edge_ids = torch.arange(3, device=device).repeat(n_tri)
    # 4. Use a unique key for each edge
    edge_keys = flat_edges[:, 0].to(torch.int64) * (simplices.max()+1) + flat_edges[:, 1].to(torch.int64)
    unique_edges, inverse_indices, counts = torch.unique(edge_keys, return_inverse=True, return_counts=True)
    # 5. Find all triangles corresponding to each edge
    mask = counts[inverse_indices] == 2
    idxs = torch.nonzero(mask, as_tuple=False).flatten()
    # 6. For each edge with neighbors, find its two triangles
    sort_idx = torch.argsort(inverse_indices[idxs])
    idxs_sorted = idxs[sort_idx]
    # Group every two
    t0 = tri_ids[idxs_sorted][0::2].to(torch.int32)
    e0 = edge_ids[idxs_sorted][0::2].to(torch.int32)
    t1 = tri_ids[idxs_sorted][1::2].to(torch.int32)
    e1 = edge_ids[idxs_sorted][1::2].to(torch.int32)
    # 7. Construct adjacency matrix
    neighbors = torch.arange(n_tri, device=device).unsqueeze(1).repeat(1, 3).to(torch.int32)
    neighbors[t0, e0] = t1
    neighbors[t1, e1] = t0
    return neighbors
    
class EllipseProcess:
    def __init__(self):
        # self.cpu_num = len(os.sched_getaffinity(0))
        self.triangles = None
        # init_pool(self.cpu_num)
        
        
    def fit_ellipse_cuda(self):
        # self.triangles [N, 3, 2]
        # Add midpoints of each triangle's edges as new points
        midpoints = (self.triangles + torch.roll(self.triangles, -1, dims=1)) / 2
        points = torch.cat([self.triangles, midpoints], dim=1)  # [N, 6, 2]
        # to tensor
        fit_result = fit_ellipses(points)
        return fit_result[:, :2], fit_result[:, 2:4], fit_result[:, 4]

    # def fit_task_batch(self, start, end):
    #     centers, axes, angles = [], [], []
    #     for i in range(start, end):
    #         tri_ver = self.triangles[i]
    #         center, axis, angle = min_bounding_ellipse(tri_ver)
    #         centers.append(center)
    #         axes.append(axis)
    #         angles.append(angle)
    #     return np.array(centers), np.array(axes), np.array(angles)

    # def fit_ellipse_parallel(self):
    #     global pool
    #     task_num = len(self.triangles) // self.cpu_num
    #     results = []
    #     for i in range(self.cpu_num):
    #         start = i * task_num
    #         end = (i + 1) * task_num if i != self.cpu_num - 1 else len(self.triangles)
    #         results.append(pool.apply_async(
    #             self.fit_task_batch,
    #             args=(start, end)
    #         ))
    #     centers, axes, angles = [], [], []
    #     for res in results:
    #         center, axis, angle = res.get()
    #         centers.append(center)
    #         axes.append(axis)
    #         angles.append(angle)
    #     centers = np.concatenate(centers, axis=0)
    #     axes = np.concatenate(axes, axis=0)
    #     angles = np.concatenate(angles, axis=0)
    #     return centers, axes, angles

    @torch.no_grad()
    def post_process(self, points, H, W):
        points = add_boundary_points(points, H, W)

        # CPU version
        # tri = Delaunay(points, incremental=True)
        # simplices = tri.simplices
        
        # GPU version with cupy
        points = cp.asarray(points)
        tri = DelaunayGPU(points)
        simplices = tri.simplices
        # There are some bugs with tri.neighbors, so compute neighbors manually
        
        self.triangles = points[simplices]  # [N, 3, 2]
        self.triangles = torch.tensor(self.triangles, dtype=torch.float32).cuda()

        # cpu version
        # ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_parallel()
        # ellipses_center = torch.tensor(ellipses_center, dtype=torch.float32).cuda()
        # ellipses_size = torch.tensor(ellipses_size, dtype=torch.float32).cuda()
        # ellipses_angle = torch.tensor(ellipses_angle, dtype=torch.float32).cuda()
        
        # cuda version
        ellipses_center, ellipses_size, ellipses_angle = self.fit_ellipse_cuda()
        
        mask = ~ellipse_filter(ellipses_size)
        self.triangles = self.triangles[mask]
        ellipses_center = ellipses_center[mask]
        ellipses_size = ellipses_size[mask]
        ellipses_angle = ellipses_angle[mask]
        
        # simplices to neighbors
        simplices = torch.tensor(simplices, dtype=torch.int32).cuda()
        simplices = simplices[mask]
        self.neighbors = simplices_to_neighbors(simplices)
        
        scale_param = torch.tensor([W, H], dtype=torch.float32, device=self.triangles.device)

        self.triangles = self.triangles[:, :, :2] / scale_param * 2 - 1
        ellipses_center = ellipses_center[:, :2] / scale_param * 2 - 1
        ellipses_center = torch.clamp(ellipses_center, -0.99999, 0.99999)

        ellipses_size = ellipses_size * 0.5
        # ellipses_angle to [0, 1]
        ellipses_angle = ellipses_angle / 360
        ellipses_angle = torch.clamp(ellipses_angle, 0.00001, 0.99999)

        # neighbors = tri.neighbors
        # neighbors = neighbors_process(neighbors)
        # neighbors = torch.tensor(neighbors, dtype=torch.int32, device=self.triangles.device)

        return self.triangles, ellipses_center, ellipses_size, ellipses_angle, self.neighbors

    @torch.no_grad()
    def process(self, pf, kernel_size):
        B, H, W = pf.shape[0], pf.shape[1], pf.shape[2]
        sampled_xy = dither_image(pf, kernel_size=kernel_size)
        elements = self.post_process(sampled_xy, H, W)
        return elements


class EllipseProcessKNN:
    """
    Fully differentiable KNN-based prior extraction that operates on the GPU and
    bypasses any Delaunay triangulation or CPU computations.
    """

    def __init__(self, k=6, sample_neighbors=3):
        self.k = max(int(k), 2)
        self.sample_neighbors = max(int(sample_neighbors), 1)

    def _pad_neighbors(self, tensor, target_len):
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len]
        pad_count = target_len - tensor.shape[1]
        pad_slice = tensor[:, -1:].repeat(1, pad_count, 1)
        return torch.cat([tensor, pad_slice], dim=1)

    def _pad_indices(self, tensor, target_len):
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len]
        pad_count = target_len - tensor.shape[1]
        pad_slice = tensor[:, -1:].repeat(1, pad_count)
        return torch.cat([tensor, pad_slice], dim=1)

    def _normalize_points(self, pts, H, W):
        scale_param = torch.tensor([W, H], dtype=torch.float32, device=pts.device)
        pts = pts / scale_param * 2 - 1
        pts = torch.clamp(pts, -0.99999, 0.99999)
        return pts

    def _compute_covariance(self, neighbors):
        mean = neighbors.mean(dim=1, keepdim=True)
        centered = neighbors - mean
        denom = max(neighbors.shape[1] - 1, 1)
        cov = torch.matmul(centered.transpose(1, 2), centered) / denom
        return cov

    def process(self, pf, kernel_size):
        
        device = pf.device
        _, H, W = pf.shape
        sampled_xy = dither_image(pf, kernel_size=kernel_size)
        if sampled_xy.numel() == 0:
            raise RuntimeError("No samples generated from the position field.")
        points = sampled_xy.to(device=device, dtype=torch.float32)
        if points.shape[0] < 2:
            raise RuntimeError("Need at least two samples to compute KNN priors.")

        points = add_boundary_points_torch(points, H, W)

        dist = torch.cdist(points, points, p=2)
        k_eff = min(self.k, max(points.shape[0] - 1, 1))
        knn_idx = torch.topk(dist, k_eff + 1, dim=1, largest=False).indices[:, 1:]
        neighbors = points[knn_idx]

        cov = self._compute_covariance(neighbors)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-8)
        major = torch.sqrt(eigvals[:, 1])
        minor = torch.sqrt(eigvals[:, 0])
        prior_scales = torch.stack([major, minor], dim=1)

        major_vec = eigvecs[:, :, 1]
        angles = torch.atan2(major_vec[:, 1], major_vec[:, 0])
        angles = (angles / (2 * math.pi)) % 1.0
        angles = torch.clamp(angles, 0.00001, 0.99999)

        neighbors_subset = self._pad_neighbors(neighbors, self.sample_neighbors)
        neighbor_idx_subset = self._pad_indices(knn_idx, self.sample_neighbors).long()

        centers_norm = self._normalize_points(points, H, W)
        neighbors_norm = self._normalize_points(neighbors_subset.reshape(-1, 2), H, W)
        neighbors_norm = neighbors_norm.view(points.shape[0], self.sample_neighbors, 2)

        prior_scales = prior_scales * 0.5

        return centers_norm, neighbors_norm, prior_scales, angles, neighbor_idx_subset


class EllipseProcessSoftKNN:
    """
    Differentiable Soft-KNN Prior. 
    Calculates covariance and neighbors with soft-weighted attention,
    ensuring gradients flow through point positions and features.
    """
    def __init__(self, k=6, sample_neighbors=3, temperature=0.1):
        self.k = max(int(k), 2)
        # We sample k neighbors for the covariance calculation, 
        # but we might only return 'sample_neighbors' to the MLP to save memory.
        self.sample_neighbors = max(int(sample_neighbors), 1)
        self.temperature = temperature

    def _normalize_points(self, pts, H, W):
        # Normalize to [-1, 1] range for grid_sample compatibility
        scale_param = torch.tensor([W, H], dtype=torch.float32, device=pts.device)
        pts = pts / scale_param * 2 - 1
        return torch.clamp(pts, -0.99999, 0.99999)

    def _pad_neighbors(self, tensor, target_len):
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len]
        pad_count = target_len - tensor.shape[1]
        pad_slice = tensor[:, -1:].repeat(1, pad_count, 1)
        return torch.cat([tensor, pad_slice], dim=1)

    def _pad_indices(self, tensor, target_len):
        if tensor.shape[1] >= target_len:
            return tensor[:, :target_len]
        pad_count = target_len - tensor.shape[1]
        pad_slice = tensor[:, -1:].repeat(1, pad_count)
        return torch.cat([tensor, pad_slice], dim=1)

    def process(self, pf, kernel_size):
        """
        pf: Position Field / Probability Map [B, 1, H, W]
        kernel_size: for the dithering step
        """
        device = pf.device
        _, H, W = pf.shape
        
        # 1. Sampling (Assuming dither_image exists in your scope)
        # This step is non-differentiable regarding the *number* of points,
        # but subsequent operations are differentiable w.r.t their positions.
        sampled_xy = dither_image(pf, kernel_size=kernel_size)
        if sampled_xy.numel() == 0:
            raise RuntimeError("No samples generated from the position field.")

        points = sampled_xy.to(device=device, dtype=torch.float32)
        if points.shape[0] < 2:
            raise RuntimeError("Need at least two samples to compute KNN priors.")

        points = add_boundary_points_torch(points, H, W)
        N = points.shape[0]

        k_eff = min(self.k + 1, N)
        knn_val_list = []
        knn_idx_list = []
        knn_chunk_size = 4096
        
        # We perform the distance calculation in blocks to save VRAM
        for i in range(0, N, knn_chunk_size):
            end = min(i + knn_chunk_size, N)
            query_chunk = points[i:end] # [Batch, 2]
            
            # cdist size: [Batch_Chunk, N] -> Much smaller than [N, N]
            dist_chunk = torch.cdist(query_chunk, points, p=2) 
            
            # Immediately reduce to top-k to free memory
            chunk_val, chunk_idx = torch.topk(dist_chunk, k_eff, dim=1, largest=False)
            
            knn_val_list.append(chunk_val)
            knn_idx_list.append(chunk_idx)
            
        # Combine the chunks
        knn_val = torch.cat(knn_val_list, dim=0) # [N, k_eff]
        knn_idx = torch.cat(knn_idx_list, dim=0) # [N, k_eff]
        
        # Exclude self (first column) for the neighbor list
        # But for covariance, we often include self or just use neighbors. 
        # Let's use neighbors only for shape context.
        neighbor_dists = knn_val[:, 1:] # [N, k]
        neighbor_indices = knn_idx[:, 1:] # [N, k]
        
        # 4. Soft-Weighted Attention (The Fix)
        # Weights decay as distance increases.
        # [N, k]
        #local_scale = neighbor_dists.mean(dim=1, keepdim=True) + 1e-6
        #weights = torch.softmax(-neighbor_dists / (local_scale * self.temperature), dim=1)
        weights = torch.ones_like(neighbor_dists) / neighbor_dists.shape[1]
        
        # Gather neighbor coordinates: [N, k, 2]
        # Expand points to gather: [N, N, 2]
        neighbors = points[neighbor_indices]
        neighbors = torch.nan_to_num(neighbors, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. Weighted Mean and Covariance
        # Mean center of the neighborhood (Soft centroid)
        # sum(w * x) -> [N, 1, 2]
        #soft_mean = (neighbors * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        
        # Centered coordinates TODO eval which is better
        #centered = neighbors - soft_mean 
        centered = neighbors - points.unsqueeze(1)  # [N, k, 2]
        
        # Weighted Covariance: sum(w * (x-u)(x-u)^T)
        # [N, k, 2, 1] * [N, k, 1, 2] -> [N, k, 2, 2]
        outer_prod = torch.matmul(centered.unsqueeze(-1), centered.unsqueeze(-2))
        # Weighted sum -> [N, 2, 2]
        cov = (outer_prod * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        # 6. Eigendecomposition for Ellipse Parameters
        # Add epsilon to diagonal for stability
        cov = cov + torch.eye(2, device=device).unsqueeze(0) * 1e-6

        chunk_size = 8192
        #if cov.shape[0] > chunk_size:
        #    print(f"[EllipseProcessSoftKNN] chunking eigen solve: total={cov.shape[0]}, chunk={chunk_size}")

        eigvals_list = []
        eigvecs_list = []
        for start in range(0, cov.shape[0], chunk_size):
            end = min(start + chunk_size, cov.shape[0])
            chunk = cov[start:end]
            vals, vecs = torch.linalg.eigh(chunk)
            eigvals_list.append(vals)
            eigvecs_list.append(vecs)

        eigvals = torch.cat(eigvals_list, dim=0)
        eigvecs = torch.cat(eigvecs_list, dim=0)
        eigvals = torch.clamp(eigvals, min=1e-8)
        
        # Get major/minor axes
        major = torch.sqrt(eigvals[:, 1])
        minor = torch.sqrt(eigvals[:, 0])
        prior_scales = torch.stack([major, minor], dim=1)
        prior_scales = prior_scales * 1.0

        # Get Angle
        major_vec = eigvecs[:, :, 1] # Eigenvector corresponding to largest eigenvalue
        angles = torch.atan2(major_vec[:, 1], major_vec[:, 0])
        
        # Normalize angle to [0, 1] for the network
        angles = (angles / (2 * math.pi)) % 1.0
        angles = torch.clamp(angles, 0.00001, 0.99999)
        
        # 7. Prepare outputs for MLP
        # We need to return specific subsets for the neural net
        
        # Select closest neighbors for the "sample_points" input of the network
        # usually we just take the top 'sample_neighbors' from our soft list
        final_neighbors = self._pad_neighbors(neighbors, self.sample_neighbors)
        final_indices = self._pad_indices(neighbor_indices, self.sample_neighbors).long()
        
        # Normalize for grid sampling features
        centers_norm = self._normalize_points(points, H, W) # The actual points
        neighbors_norm = self._normalize_points(final_neighbors.reshape(-1, 2), H, W)
        neighbors_norm = neighbors_norm.view(points.shape[0], self.sample_neighbors, 2)
        
        # Note: We return 'points' as the center, but the ellipse properties 
        # are derived from the 'soft_mean'. This gives us the residuals we need.
        
        return centers_norm, neighbors_norm, prior_scales, angles, final_indices
