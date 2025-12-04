"""
Renderer module that converts Gaussian splats to images using an adapter and patch projection.

The renderer:
1. Takes in Gaussian splats
2. Passes them through a GaussianAdapter to get 196 latent tokens
3. Projects each latent token to a 16x16 pixel patch
4. Tiles all patches together to form a 224x224 image
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .adapter import GaussianAdapter


class GaussianRenderer(nn.Module):
    """
    Renderer that converts Gaussian splats to images.
    
    Pipeline:
    1. Splats -> GaussianAdapter -> 196 latent tokens (B, 196, D)
    2. Each latent token -> Linear projection -> 16x16 patch (B, 196, 16*16*3)
    3. Reshape and tile patches -> 224x224 image (B, 3, 224, 224)
    """
    
    def __init__(
        self,
        latent_seq_len: int = 196,
        embed_dim: int = 768,
        num_heads: int = 8,
        patch_size: int = 16,
        image_size: int = 224,
        adapter_config: Optional[Dict] = None,
        latent_init_std: float = 0.02,
    ):
        """
        Initialize GaussianRenderer.
        
        Args:
            latent_seq_len: Number of latent tokens (default: 196, matching 14x14 patches).
            embed_dim: Embedding dimension (default: 768).
            num_heads: Number of attention heads for adapter cross-attention (default: 8).
            patch_size: Size of each patch in pixels (default: 16).
            image_size: Size of output image in pixels (default: 224).
            adapter_config: Optional dict of additional arguments for GaussianAdapter.
                          If None, uses defaults.
            latent_init_std: Standard deviation for latent vector initialization (default: 0.02).
        """
        super().__init__()
        
        self.latent_seq_len = latent_seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Verify dimensions match
        num_patches_per_side = image_size // patch_size
        expected_latent_seq_len = num_patches_per_side * num_patches_per_side
        if latent_seq_len != expected_latent_seq_len:
            raise ValueError(
                f"latent_seq_len ({latent_seq_len}) must equal "
                f"(image_size // patch_size)^2 = {expected_latent_seq_len}"
            )
        
        # Initialize adapter
        adapter_kwargs = adapter_config or {}
        adapter_kwargs.setdefault('fourier_freqs', 6)
        adapter_kwargs.setdefault('output_dim', embed_dim)
        adapter_kwargs.setdefault('fourier_dim', 24)
        adapter_kwargs.setdefault('num_heads', num_heads)
        adapter_kwargs.setdefault('latent_seq_len', latent_seq_len)
        adapter_kwargs.setdefault('cross_attn_dropout', 0.1)
        adapter_kwargs.setdefault('use_cross_attention', True)
        
        self.adapter = GaussianAdapter(**adapter_kwargs)
        
        # Initialize learnable latent vectors: (latent_seq_len, embed_dim)
        self.latent_vectors = nn.Parameter(
            torch.randn(latent_seq_len, embed_dim) * latent_init_std
        )
        
        # Projection layer: each latent token -> patch of pixels
        # Each patch is patch_size x patch_size x 3 (RGB)
        patch_pixels = patch_size * patch_size * 3
        self.patch_projection = nn.Linear(embed_dim, patch_pixels)
        
        # Initialize patch projection
        nn.init.normal_(self.patch_projection.weight, std=0.02)
        if self.patch_projection.bias is not None:
            nn.init.zeros_(self.patch_projection.bias)
    
    def forward(
        self,
        splat_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the renderer.
        
        Args:
            splat_data: Dictionary containing splat components (batched or unbatched):
                - 'xy': Tensor of shape [B, M, 2] or [M, 2] - x,y coordinates
                - 'scaling': Tensor of shape [B, M, 2] or [M, 2] - scaling parameters
                - 'rotation': Tensor of shape [B, M, 1] or [M, 1] - rotation parameter
                - 'color': Tensor of shape [B, M, 3] or [M, 3] - color values
            attention_mask: Optional tensor of shape [B, M] or [M] - 1/True if real splat,
                          0/False if padding (boolean or integer tensor).
        
        Returns:
            Tensor of shape [B, 3, image_size, image_size] or [3, image_size, image_size] - rendered image
        """
        # Get batch size and device
        xy = splat_data['xy']
        was_unbatched = xy.dim() == 2
        
        if was_unbatched:
            xy = xy.unsqueeze(0)
        
        B = xy.shape[0]
        device = xy.device
        
        # Expand latent vectors to batch size: (latent_seq_len, embed_dim) -> (B, latent_seq_len, embed_dim)
        latent_vectors = self.latent_vectors.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        
        # Pass through adapter: splats -> updated latent vectors
        # Returns (B, L, D) or (L, D) if unbatched
        latent_tokens = self.adapter(
            splat_data=splat_data,
            latent_vectors=latent_vectors,
            attention_mask=attention_mask,
        )  # (B, L, D) or (L, D)
        
        # Ensure batched for processing
        if was_unbatched:
            latent_tokens = latent_tokens.unsqueeze(0)
            B = 1
        
        # Project each latent token to a patch: (B, L, D) -> (B, L, patch_size * patch_size * 3)
        patches_flat = self.patch_projection(latent_tokens)  # (B, L, patch_size^2 * 3)
        
        # Reshape patches: (B, L, patch_size^2 * 3) -> (B, L, patch_size, patch_size, 3)
        num_patches_per_side = self.image_size // self.patch_size
        patches = patches_flat.view(
            B, num_patches_per_side, num_patches_per_side, 
            self.patch_size, self.patch_size, 3
        )  # (B, H_patches, W_patches, patch_size, patch_size, 3)
        
        # Permute to group spatial dimensions: (B, H_patches, W_patches, patch_size, patch_size, 3)
        # -> (B, H_patches, patch_size, W_patches, patch_size, 3)
        patches = patches.permute(0, 1, 3, 2, 4, 5)  # (B, H_patches, patch_size, W_patches, patch_size, 3)
        
        # Reshape to combine patch dimensions: (B, H_patches, patch_size, W_patches, patch_size, 3)
        # -> (B, image_size, image_size, 3)
        image = patches.contiguous().view(B, self.image_size, self.image_size, 3)
        
        # Permute to CHW format: (B, H, W, 3) -> (B, 3, H, W)
        image = image.permute(0, 3, 1, 2)  # (B, 3, image_size, image_size)
        
        # Clamp to valid pixel range [0, 1]
        image = torch.clamp(image, 0.0, 1.0)
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            image = image.squeeze(0)  # (3, image_size, image_size)
        
        return image


def create_renderer(
    latent_seq_len: int = 196,
    embed_dim: int = 768,
    num_heads: int = 8,
    patch_size: int = 16,
    image_size: int = 224,
    **kwargs
) -> GaussianRenderer:
    """
    Factory function to create a GaussianRenderer.
    
    Args:
        latent_seq_len: Number of latent tokens (default: 196).
        embed_dim: Embedding dimension (default: 768).
        num_heads: Number of attention heads (default: 8).
        patch_size: Size of each patch in pixels (default: 16).
        image_size: Size of output image in pixels (default: 224).
        **kwargs: Additional arguments passed to GaussianRenderer.
    
    Returns:
        GaussianRenderer instance
    """
    return GaussianRenderer(
        latent_seq_len=latent_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        image_size=image_size,
        **kwargs
    )

