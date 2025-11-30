import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


def _resolve_pipeline_device(device: Optional[Union[str, int, torch.device]]) -> Union[int, str]:
    """Returns a device hint that HuggingFace's pipeline understands."""
    if isinstance(device, torch.device):
        if device.type == "cpu":
            return -1
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return device.type
    if isinstance(device, int):
        return device
    if device is None or device == "auto":
        return 0 if torch.cuda.is_available() else -1
    if isinstance(device, str):
        lowered = device.lower()
        if lowered == "cpu":
            return -1
        if lowered.startswith("cuda"):
            if ":" in lowered:
                return int(lowered.split(":")[1])
            return 0
        return lowered
    raise ValueError(f"Unsupported device specifier: {device}")


@dataclass
class DepthModelConfig:
    """Configuration bundle for the HuggingFace depth-estimation pipeline."""

    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: Optional[Union[str, int, torch.device]] = "auto"
    cache_dir: Optional[str] = None
    dtype: torch.dtype = torch.float32


class DepthEstimator:
    """Wraps a HuggingFace depth-estimation pipeline and exposes a simple API."""

    def __init__(self, config: DepthModelConfig):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Depth estimation requires the 'transformers' package. "
                "Install it with `pip install transformers`."
            ) from exc

        hf_device = _resolve_pipeline_device(config.device)
        self._pipeline = pipeline(
            task="depth-estimation",
            model=config.model_id,
            device=hf_device,
            cache_dir=config.cache_dir,
        )
        self._dtype = config.dtype

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Returns a depth map for an HxWx3 float tensor in [0, 1]."""
        if image.dim() != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"DepthEstimator expects an HxWx3 tensor, got shape {tuple(image.shape)}."
            )
        image_uint8 = (
            image.detach()
            .to(torch.float32)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        pil_image = Image.fromarray(image_uint8, mode="RGB")
        result: Dict[str, Any] = self._pipeline(pil_image)
        depth_image = result.get("depth")
        if depth_image is None:
            raise RuntimeError("Depth pipeline did not return a 'depth' map.")
        depth = torch.from_numpy(np.array(depth_image, dtype=np.float32))
        return depth.to(self._dtype)


def create_depth_estimator(
    model_id: str,
    device: Optional[Union[str, int, torch.device]] = "auto",
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> DepthEstimator:
    """Convenience helper to instantiate a depth estimator."""
    config = DepthModelConfig(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype)
    return DepthEstimator(config)

