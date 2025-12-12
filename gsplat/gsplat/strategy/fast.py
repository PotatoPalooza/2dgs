from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from .default import DefaultStrategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class FastStrategy(DefaultStrategy):
    """Fast-GS-style multi-view consistent densification/pruning.

    This strategy approximates the Fast-GS paper (arXiv:2511.04283) while
    operating purely on Gaussian-level signals available in gsplat.

    It adds two multi-view scores:
      - VCD score: avg. count of high-error pixels inside each Gaussian's 2D footprint.
      - VCP score: same count weighted by per-view photometric loss.

    Both scores are accumulated across views (steps) and used to gate growth
    and guide pruning, reducing redundant Gaussians.

    Args:
        high_err_tau (float): Threshold on min-max normalized per-pixel L1 map
            to define a high-error mask (tau in paper).
        vcd_thresh (float): Densification threshold tau_d (paper default 5).
        vcd_min_views (int): Minimum accumulated views before trusting VCD.
        vcp_thresh (float): Pruning threshold tau_p after refine_stop_iter (paper 0.9).
        vcp_pre_fraction (float): Fraction of baseline prune candidates to keep
            before refine_stop_iter, based on highest VCP scores (paper uses ~0.5).
        mv_decay (float): EMA for multi-view stats (1.0 disables EMA).
        vcd_sample_points (int): Number of sample points per footprint to estimate
            high-error overlap (default 9).
    """

    # Defaults: tuned to be less aggressive than the paper defaults.
    # The Fast-GS paper settings (tau=0.2, vcd_thresh=5, vcd_min_views=2, etc.)
    # can under-densify early (VCD gate too strict) while still pruning via
    # baseline opacity/size pruning. These defaults aim to:
    # - allow growth earlier (lower VCD thresholds),
    # - reduce pre-stop pruning pressure (lower vcp_pre_fraction),
    # while keeping the overall spirit of multi-view pruning.
    high_err_tau: float = 0.1
    vcd_thresh: float = 2.0
    vcd_min_views: int = 1
    vcp_thresh: float = 0.97
    vcp_pre_fraction: float = 0.2
    # Paper-style accumulation across views (no EMA). If you enable EMA by setting
    # mv_decay < 1, min-views gating uses a separate raw counter.
    mv_decay: float = 1.0
    vcd_sample_points: int = 9

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update(
            {
                "vcd_sum": None,
                "vcd_count": None,
                "vcp_sum": None,
                "vcp_count": None,
                "vcd_views": None,
                "vcp_views": None,
            }
        )
        return state

    def _minmax_norm(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        x_min = x.min()
        x_max = x.max()
        denom = (x_max - x_min).clamp_min(1e-8)
        return (x - x_min) / denom

    def _update_state(  # type: ignore[override]
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        super()._update_state(params, state, info, packed=packed)

        if "high_err_mask" not in info or "means2d" not in info:
            return

        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        coords = info["means2d"].detach()
        radii2d = info["radii"].detach()
        high_err_mask = info["high_err_mask"].detach()
        photo_loss = info.get("photo_loss", None)
        if photo_loss is not None:
            photo_loss = photo_loss.detach()

        width, height = int(info["width"]), int(info["height"])
        n_gaussian = len(list(params.values())[0])
        device = coords.device
        if state["vcd_sum"] is None:
            state["vcd_sum"] = torch.zeros(n_gaussian, device=device)
        if state["vcd_count"] is None:
            state["vcd_count"] = torch.zeros(n_gaussian, device=device)
        if state["vcp_sum"] is None:
            state["vcp_sum"] = torch.zeros(n_gaussian, device=device)
        if state["vcp_count"] is None:
            state["vcp_count"] = torch.zeros(n_gaussian, device=device)
        if state.get("vcd_views") is None:
            state["vcd_views"] = torch.zeros(n_gaussian, device=device)
        if state.get("vcp_views") is None:
            state["vcp_views"] = torch.zeros(n_gaussian, device=device)

        if packed:
            gs_ids = info["gaussian_ids"]
            # Packed mode uses explicit camera indices for each projected gaussian.
            # `image_ids` is not guaranteed to be present in gsplat's packed info dict.
            view_ids = info.get("camera_ids", None)
            coords_vis = coords
            radii_vis = radii2d.max(dim=-1).values
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)
            where = torch.where(sel)
            view_ids = where[0]
            gs_ids = where[1]
            coords_vis = coords[sel]
            radii_vis = radii2d[sel].max(dim=-1).values

        if coords_vis.numel() == 0:
            return

        # Approximate high-error overlap by sampling a fixed grid in each 2D footprint.
        # Sample pattern: center + 8 points on a 3x3 grid.
        if self.vcd_sample_points <= 1:
            offsets = torch.zeros((1, 2), device=device)
        else:
            offsets = torch.tensor(
                [
                    [0.0, 0.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                device=device,
                dtype=coords_vis.dtype,
            )
            offsets = offsets[: self.vcd_sample_points]

        centers = coords_vis  # [nnz, 2] in pixel coordinates
        r = radii_vis.clamp_min(1.0)  # [nnz]
        samples = centers[:, None, :] + offsets[None, :, :] * r[:, None, None]
        xs = samples[..., 0].round().long().clamp(0, width - 1)
        ys = samples[..., 1].round().long().clamp(0, height - 1)

        if view_ids is None:
            # Single view; high_err_mask is [H, W] or [1, H, W]
            mask_view = high_err_mask
            if mask_view.dim() == 3:
                mask_view = mask_view[0]
            sampled = mask_view[ys, xs]
            counts = sampled.float().sum(dim=-1)
            photo_w = None
            if photo_loss is not None:
                photo_w = photo_loss.reshape(-1)[0]
        else:
            # high_err_mask is [C, H, W]
            sampled = high_err_mask[view_ids[:, None], ys, xs]
            counts = sampled.float().sum(dim=-1)
            photo_w = None
            if photo_loss is not None:
                photo_w = photo_loss[view_ids]

        if self.mv_decay < 1.0:
            decay = self.mv_decay
            keep = 1.0 - decay
            state["vcd_sum"][gs_ids] = decay * state["vcd_sum"][gs_ids] + keep * counts
            state["vcd_count"][gs_ids] = (
                decay * state["vcd_count"][gs_ids] + keep * 1.0
            )
            # Maintain raw view counters for min-view gating.
            state["vcd_views"].index_add_(
                0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
            )
            if photo_w is not None:
                weighted = counts * photo_w
                state["vcp_sum"][gs_ids] = (
                    decay * state["vcp_sum"][gs_ids] + keep * weighted
                )
                state["vcp_count"][gs_ids] = (
                    decay * state["vcp_count"][gs_ids] + keep * 1.0
                )
                state["vcp_views"].index_add_(
                    0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
                )
        else:
            state["vcd_sum"].index_add_(0, gs_ids, counts)
            state["vcd_count"].index_add_(
                0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
            )
            state["vcd_views"].index_add_(
                0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
            )
            if photo_w is not None:
                state["vcp_sum"].index_add_(0, gs_ids, counts * photo_w)
                state["vcp_count"].index_add_(
                    0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
                )
                state["vcp_views"].index_add_(
                    0, gs_ids, torch.ones_like(counts, dtype=torch.float32)
                )

    def _vcd_score(self, state: Dict[str, Any]) -> Union[torch.Tensor, None]:
        if state.get("vcd_sum") is None or state.get("vcd_count") is None:
            return None
        return state["vcd_sum"] / state["vcd_count"].clamp_min(1.0)

    def _vcp_score(self, state: Dict[str, Any]) -> Union[torch.Tensor, None]:
        if state.get("vcp_sum") is None or state.get("vcp_count") is None:
            return None
        return state["vcp_sum"] / state["vcp_count"].clamp_min(1.0)

    @torch.no_grad()
    def step_post_backward(  # type: ignore[override]
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            if state.get("vcd_sum") is not None:
                state["vcd_sum"].zero_()
                state["vcd_count"].zero_()
                if state.get("vcd_views") is not None:
                    state["vcd_views"].zero_()
            if state.get("vcp_sum") is not None:
                state["vcp_sum"].zero_()
                state["vcp_count"].zero_()
                if state.get("vcp_views") is not None:
                    state["vcp_views"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    @torch.no_grad()
    def _grow_gs(  # type: ignore[override]
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ):
        vcd = self._vcd_score(state)
        vcd_mask = None
        if vcd is not None:
            views = state.get("vcd_views", None)
            if views is None:
                views = state["vcd_count"]
            vcd_mask = (vcd > self.vcd_thresh) & (
                views >= float(self.vcd_min_views)
            )

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        if vcd_mask is not None:
            is_dupli &= vcd_mask
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        if vcd_mask is not None:
            is_split &= vcd_mask
        n_split = is_split.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(  # type: ignore[override]
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ):
        opacities = torch.sigmoid(params["opacities"].flatten())
        baseline_prune = opacities < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            baseline_prune = baseline_prune | is_too_big

        vcp = self._vcp_score(state)
        if vcp is not None:
            vcp_norm = self._minmax_norm(vcp)
        else:
            vcp_norm = None

        is_prune = baseline_prune

        # Pre-15k: prune top fraction of baseline candidates by VCP.
        if vcp_norm is not None and step < self.refine_stop_iter:
            candidates = baseline_prune
            if candidates.any():
                scores_c = vcp_norm[candidates]
                q = torch.quantile(
                    scores_c, 1.0 - float(self.vcp_pre_fraction)
                ).item()
                is_prune = candidates & (vcp_norm >= q)

        # Post-15k: prune by high VCP score as in paper.
        if vcp_norm is not None and step >= self.refine_stop_iter:
            is_prune = is_prune | (vcp_norm > self.vcp_thresh)

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
