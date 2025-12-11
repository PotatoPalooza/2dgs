"""
Utilities for loading 2D Gaussian parameters saved by Instant-GI.

The Instant-GI training loop stores a checkpoint named ``gaussian_model.pth.tar``
that contains the raw learnable tensors from :class:`GaussianImage_RS`:

    - ``_xyz``: (N, 2) unconstrained xy locations
    - ``_scaling``: (N, 2) unconstrained scale values
    - ``_rotation``: (N, 1) unconstrained rotation values
    - ``_opacity``: (N, 1) opacity values
    - ``_features_dc``: (N, 3) RGB colors
    - ``bound``: (1, 2) helper tensor used when sampling random init points

Scenes exported with Instant-GI are organized as:

    <scene_root>/
        <frame_name>/
            net/gaussian_model.pth.tar     # when initialized from the InitNet
            random/gaussian_model.pth.tar  # when initialized randomly (optional)

This module provides a small Dataset wrapper that discovers those files,
applies the same activations used inside Instant-GI (tanh on xy, relu+clamp
or abs+bound on scale, sigmoid*2π on rotation), and returns ready-to-use
tensors for xy/scale/rotation/opacity/color and a stacked (N, 9) array.

Note: This code intentionally re-implements the small conversion logic
instead of importing Instant-GI so it can run inside the gsplat conda env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

InitMethod = Literal["net", "random"]


def _apply_instant_gi_activation(
    xy: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    init_method: InitMethod,
    bound: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same parameter activations used in Instant-GI."""
    xy = torch.tanh(xy)

    if init_method == "net":
        scale = torch.clamp(torch.relu(scale), min=0.5)
    elif init_method == "random":
        default_bound = torch.tensor(
            [0.5, 0.5], dtype=scale.dtype, device=scale.device
        ).view(1, 2)
        scale = torch.abs(scale + (default_bound if bound is None else bound))
    else:
        raise ValueError(f"Unknown init_method '{init_method}'")

    rotation = torch.sigmoid(rotation) * 2 * math.pi

    return xy, scale, rotation, opacity, color


def _load_gaussian_checkpoint(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load checkpoint safely with weights_only when available."""
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(model_path, weights_only=True, **load_kwargs)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(model_path, **load_kwargs)


def load_gaussian_model(
    model_path: Union[str, Path],
    init_method: InitMethod = "net",
    apply_activation: bool = True,
    as_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, "np.ndarray"]]:
    """
    Load and unpack a gaussian_model.pth.tar file saved by Instant-GI.

    Args:
        model_path: Path to gaussian_model.pth.tar.
        init_method: How the model was initialized (``"net"`` or ``"random"``)
            which controls the scale activation.
        apply_activation: If True, apply Instant-GI activations to produce
            usable parameters; if False, return the raw stored tensors.
        as_numpy: If True, return numpy arrays instead of torch tensors.

    Returns:
        Dictionary with xy, scale, rotation, opacity, color, and gaussians
        (stacked [x, y, sx, sy, rot, opacity, r, g, b]).
    """
    model_path = Path(model_path)
    checkpoint = _load_gaussian_checkpoint(model_path)

    xy = checkpoint["_xyz"]
    scale = checkpoint["_scaling"]
    rotation = checkpoint["_rotation"]
    opacity = checkpoint.get("_opacity", torch.ones_like(rotation))
    color = checkpoint["_features_dc"]
    bound = checkpoint.get("bound")

    if apply_activation:
        xy, scale, rotation, opacity, color = _apply_instant_gi_activation(
            xy, scale, rotation, opacity, color, init_method, bound
        )

    gaussians = torch.cat([xy, scale, rotation, opacity, color], dim=1)

    if as_numpy:
        import numpy as np

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


@dataclass
class GaussianFrame:
    """Metadata for a single frame in a 2D Gaussian scene."""

    name: str
    path: Path
    init_method: InitMethod
    num_gaussians: int


class GaussianImageDataset(Dataset):
    """
    Torch Dataset that enumerates Instant-GI ``gaussian_model.pth.tar`` files.

    Point it at a scene directory (e.g. ``examples/data/360_v2_gs/garden_d4``)
    and it will recursively find every ``gaussian_model.pth.tar`` file that
    lives under ``<frame>/<init_method>/``. Each item returns a dictionary with
    activated xy/scale/rotation/opacity/color tensors plus a stacked ``gaussians``
    tensor for convenience alongside the frame name.
    """

    def __init__(
        self,
        scene_root: Union[str, Path],
        init_method: InitMethod = "net",
        apply_activation: bool = True,
        as_numpy: bool = False,
    ) -> None:
        self.scene_root = Path(scene_root)
        self.init_method = init_method
        self.apply_activation = apply_activation
        self.as_numpy = as_numpy

        self.frames: List[GaussianFrame] = []
        self.frames_by_name: Dict[str, GaussianFrame] = {}
        self._discover_frames()

        if len(self.frames) == 0:
            raise FileNotFoundError(
                f"No gaussian_model.pth.tar files found under {self.scene_root}"
            )

    def _discover_frames(self) -> None:
        """Find gaussian_model.pth.tar files and record their sizes."""
        # We intentionally keep this simple: look for <anything>/<init_method>/gaussian_model.pth.tar.
        pattern = f"**/{self.init_method}/gaussian_model.pth.tar"
        for model_path in sorted(self.scene_root.glob(pattern)):
            name = model_path.parent.parent.name  # frame directory name
            checkpoint = _load_gaussian_checkpoint(model_path)
            num_gaussians = checkpoint["_xyz"].shape[0]
            frame = GaussianFrame(
                name=name,
                path=model_path,
                init_method=self.init_method,
                num_gaussians=num_gaussians,
            )
            self.frames.append(frame)
            self.frames_by_name[name] = frame

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor, "np.ndarray"]]:
        frame = self.frames[idx]
        params = load_gaussian_model(
            frame.path,
            init_method=frame.init_method,
            apply_activation=self.apply_activation,
            as_numpy=self.as_numpy,
        )
        return {"name": frame.name, **params}

    @property
    def total_gaussians(self) -> int:
        """Total number of 2D Gaussians across the dataset."""
        return sum(frame.num_gaussians for frame in self.frames)

    def summary(self) -> str:
        """Human-readable overview of the dataset."""
        return (
            f"{len(self.frames)} frames from {self.scene_root} "
            f"(init={self.init_method}, total splats={self.total_gaussians})"
        )

    def get_frame(self, name: str) -> Optional[GaussianFrame]:
        """Return frame metadata by name (e.g., image stem like DSC07956)."""
        return self.frames_by_name.get(name)

    def get_gaussians_for_image(self, image_path: Union[str, Path]):
        """
        Load parameters for the frame matching an image filename.

        Args:
            image_path: Path to an image; its stem should match the frame dir.
        """
        stem = Path(image_path).stem
        frame = self.get_frame(stem)
        if frame is None:
            raise KeyError(f"No gaussian frame found matching image '{stem}'")
        return load_gaussian_model(
            frame.path,
            init_method=frame.init_method,
            apply_activation=self.apply_activation,
            as_numpy=self.as_numpy,
        )
