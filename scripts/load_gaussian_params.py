#!/usr/bin/env python3
"""
Utilities for reading Instant-GI gaussian_model.pth.tar checkpoints and
returning per-Gaussian parameters in a convenient format.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import torch


InitMethod = Literal["net", "random"]


@torch.no_grad()
def load_gaussian_parameters(
    model_path: Union[str, Path],
    init_method: InitMethod = "net",
    apply_activation: bool = True,
    as_numpy: bool = True,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Load a gaussian_model.pth.tar saved from Instant-GI and unpack the per-point
    parameters.

    Args:
        model_path: Path to gaussian_model.pth.tar.
        init_method: How the model was initialized. "net" uses the activation
            from GaussianImage_RS when init_points are provided
            (relu + clamp(min=0.5)). "random" uses the activation from random
            initialization (abs(_scaling + bound)).
        apply_activation: If True, applies the same activations used by
            GaussianImage_RS (tanh on xy, rotation sigmoid * 2pi, scaling as
            described above). If False, returns the raw stored tensors.
        as_numpy: If True, outputs numpy arrays; otherwise, torch tensors.

    Returns:
        Dict with:
            - xy: (N, 2) gaussian centers in normalized [-1, 1] if activated.
            - scale: (N, 2) gaussian scales after activation.
            - rotation: (N, 1) radians in [0, 2pi] if activated.
            - opacity: (N, 1) opacity values.
            - color: (N, 3) RGB features.
            - gaussians: (N, 9) stacked parameters in order
              [x, y, sx, sy, rot, opacity, r, g, b].
    """
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location="cpu")

    xy = checkpoint["_xyz"]
    scale = checkpoint["_scaling"]
    rotation = checkpoint["_rotation"]
    opacity = checkpoint.get("_opacity", torch.ones_like(rotation))
    color = checkpoint["_features_dc"]

    if apply_activation:
        xy = torch.tanh(xy)

        if init_method == "net":
            scale = torch.clamp(torch.relu(scale), min=0.5)
        elif init_method == "random":
            bound = checkpoint.get(
                "bound", torch.tensor([0.5, 0.5], dtype=scale.dtype, device=scale.device)
            ).view(1, 2)
            scale = torch.abs(scale + bound)
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

        rotation = torch.sigmoid(rotation) * 2 * math.pi

    gaussians = torch.cat([xy, scale, rotation, opacity, color], dim=1)

    if as_numpy:
        return {
            "xy": xy.cpu().numpy(),
            "scale": scale.cpu().numpy(),
            "rotation": rotation.cpu().numpy(),
            "opacity": opacity.cpu().numpy(),
            "color": color.cpu().numpy(),
            "gaussians": gaussians.cpu().numpy(),
        }

    return {
        "xy": xy,
        "scale": scale,
        "rotation": rotation,
        "opacity": opacity,
        "color": color,
        "gaussians": gaussians,
    }

