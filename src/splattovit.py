"""
SplatToViT: A model that converts Gaussian splats to ViT-compatible tokens.

The model consists of:
1. GaussianAdapter: Converts splats to features and applies cross-attention with latent vectors
2. Learnable latent vectors: 196 tokens (matching ViT patch tokens)
3. ViT's pretrained CLS token and positional embeddings (frozen)
4. ViT model: Processes tokens with pretrained positional embeddings
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .adapter import GaussianAdapter


class SplatToViT(nn.Module):
    """
    Model that converts Gaussian splats to ViT tokens and processes them through a ViT.
    
    The pipeline:
    1. Splats -> Adapter (with cross-attention to latent vectors) -> Updated latent vectors (B, L, D)
    2. Add positional embeddings to latent tokens (positions 1 to L)
    3. Get ViT's CLS token and add positional embedding (position 0)
    4. Concatenate CLS token and latent tokens -> (B, L+1, D)
    5. Process through ViT transformer blocks -> (B, L+1, D)
    """
    
    def __init__(
        self,
        latent_seq_len: int = 196,
        embed_dim: int = 768,
        num_heads: int = 8,
        adapter_config: Optional[Dict] = None,
        vit_model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,
        latent_init_std: float = 0.02,
    ):
        """
        Initialize SplatToViT model.
        
        Args:
            latent_seq_len: Number of latent tokens (default: 196, matching 14x14 patches).
            embed_dim: Embedding dimension (default: 768 for ViT-base).
            num_heads: Number of attention heads for adapter cross-attention (default: 8).
            adapter_config: Optional dict of additional arguments for GaussianAdapter.
                          If None, uses defaults.
            vit_model_name: Name of ViT model from timm (default: "vit_base_patch16_224").
            pretrained: Whether to use pretrained ViT weights (default: True).
            num_classes: Number of output classes (default: 1000 for ImageNet).
            latent_init_std: Standard deviation for latent vector initialization (default: 0.02).
            Note: cls_token_init_std is no longer used (ViT's pretrained CLS token is used instead).
        """
        super().__init__()
        
        self.latent_seq_len = latent_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
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
        
        # Load ViT model
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for SplatToViT. Install it with: pip install timm"
            )
        
        # Create ViT model without classification head (num_classes=0)
        # We'll use our own learnable head instead
        self.vit = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove pretrained head
            img_size=None,  # We're not using image input
        )
        
        # Extract ViT's CLS token and positional embeddings
        self._extract_vit_embeddings()

        # Freeze ViT parameters (including CLS token and positional embeddings)
        for p in self.vit.parameters():
            p.requires_grad = False

        
        # Verify embed_dim matches
        if hasattr(self.vit, 'embed_dim'):
            vit_embed_dim = self.vit.embed_dim
        elif hasattr(self.vit, 'blocks') and len(self.vit.blocks) > 0:
            vit_embed_dim = self.vit.blocks[0].norm1.normalized_shape[0]
        else:
            # Try to infer from patch embedding
            if hasattr(self.vit, 'patch_embed'):
                vit_embed_dim = self.vit.patch_embed.proj.out_channels
            else:
                vit_embed_dim = embed_dim  # Assume match
        
        if vit_embed_dim != embed_dim:
            raise ValueError(
                f"ViT embed_dim ({vit_embed_dim}) does not match adapter embed_dim ({embed_dim}). "
                f"Please ensure they match or modify the ViT embedding layer."
            )
        
        # Create our own learnable classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def _extract_vit_embeddings(self):
        """
        Extract ViT's CLS token and positional embeddings.
        
        ViT typically has:
        - cls_token: (1, embed_dim) - learnable CLS token
        - pos_embed: (1, num_patches + 1, embed_dim) - positional embeddings
          where position 0 is for CLS token, positions 1 to num_patches+1 are for patches
        """
        # Extract CLS token
        if not hasattr(self.vit, 'cls_token'):
            raise AttributeError(
                "ViT model does not have 'cls_token' attribute. "
                "Please ensure you're using a standard Vision Transformer from timm."
            )
        
        # Extract positional embeddings
        if not hasattr(self.vit, 'pos_embed'):
            raise AttributeError(
                "ViT model does not have 'pos_embed' attribute. "
                "Please ensure you're using a standard Vision Transformer from timm."
            )
        
        # pos_embed shape is typically (1, num_patches + 1, embed_dim)
        # where position 0 is for CLS token, positions 1 to num_patches+1 are for patches
        pos_embed = self.vit.pos_embed  # (1, N+1, D) where N is num_patches
        
        # Verify we have enough positional embeddings
        if pos_embed.shape[1] < self.latent_seq_len + 1:
            raise ValueError(
                f"ViT positional embeddings have {pos_embed.shape[1]} positions, "
                f"but we need {self.latent_seq_len + 1} positions (1 for CLS + {self.latent_seq_len} for latent tokens)."
            )
        
        # Extract positional embedding for CLS token (position 0)
        self.vit_cls_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, D)
        
        # Extract positional embeddings for latent tokens (positions 1 to latent_seq_len)
        self.vit_latent_pos_embed = pos_embed[:, 1:self.latent_seq_len+1, :]  # (1, L, D)
        
        # Store CLS token reference
        self.vit_cls_token = self.vit.cls_token  # (1, 1, D) or (1, D)
    
    def forward(
        self,
        splat_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through SplatToViT.
        
        Args:
            splat_data: Dictionary containing splat components (batched or unbatched):
                - 'xy': Tensor of shape [B, M, 2] or [M, 2] - x,y coordinates
                - 'scaling': Tensor of shape [B, M, 2] or [M, 2] - scaling parameters
                - 'rotation': Tensor of shape [B, M, 1] or [M, 1] - rotation parameter
                - 'color': Tensor of shape [B, M, 3] or [M, 3] - color values
            attention_mask: Optional tensor of shape [B, M] or [M] - 1/True if real splat,
                          0/False if padding (boolean or integer tensor).
        
        Returns:
            Tensor of shape [B, num_classes] or [num_classes] - class logits
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
        
        # Move positional embeddings and CLS token to the correct device
        latent_pos_embed = self.vit_latent_pos_embed.to(device)  # (1, L, D)
        cls_pos_embed = self.vit_cls_pos_embed.to(device)  # (1, 1, D)
        vit_cls_token = self.vit_cls_token.to(device)  # (1, D) or (1, 1, D)
        
        # Add positional embeddings to latent tokens (positions 1 to L)
        # vit_latent_pos_embed: (1, L, D) -> expand to (B, L, D)
        latent_pos_embed = latent_pos_embed.expand(B, -1, -1)  # (B, L, D)
        latent_tokens_with_pos = latent_tokens + latent_pos_embed  # (B, L, D)
        
        # Get ViT's CLS token and add positional embedding (position 0)
        # vit_cls_token is typically (1, embed_dim) in timm, need to add sequence dimension
        # to get (1, 1, embed_dim)
        if vit_cls_token.dim() == 2:
            # Shape is (1, D), add sequence dimension: (1, D) -> (1, 1, D)
            vit_cls_token = vit_cls_token.unsqueeze(1)
        else:
            # Already has sequence dimension, ensure it's (1, 1, D)
            vit_cls_token = vit_cls_token.view(1, 1, -1)
        
        # Expand CLS token to batch size: (1, 1, D) -> (B, 1, D)
        cls_tokens = vit_cls_token.expand(B, -1, -1)  # (B, 1, D)
        
        # Add positional embedding for CLS token (position 0)
        # vit_cls_pos_embed: (1, 1, D) -> expand to (B, 1, D)
        cls_pos_embed = cls_pos_embed.expand(B, -1, -1)  # (B, 1, D)
        cls_tokens_with_pos = cls_tokens + cls_pos_embed  # (B, 1, D)
        
        # Concatenate CLS token and latent tokens: (B, 1, D) + (B, L, D) -> (B, L+1, D)
        tokens = torch.cat([cls_tokens_with_pos, latent_tokens_with_pos], dim=1)  # (B, L+1, D)
        
        # Process through ViT transformer blocks
        # The ViT expects input of shape (B, N, D) where N is sequence length
        # We bypass the patch embedding step (already have tokens with positional embeddings)
        # and directly process tokens through transformer blocks
        
        x = tokens  # (B, L+1, D)
        
        # Apply norm before blocks if it exists (some ViT variants have this)
        if hasattr(self.vit, 'norm_pre'):
            x = self.vit.norm_pre(x)
        
        # Pass through transformer blocks
        if hasattr(self.vit, 'blocks'):
            for block in self.vit.blocks:
                x = block(x)
        else:
            raise AttributeError(
                "ViT model does not have 'blocks' attribute. "
                "Please ensure you're using a standard Vision Transformer from timm."
            )
        
        # Apply final norm (most ViTs have this)
        if hasattr(self.vit, 'norm'):
            x = self.vit.norm(x)
        
        # Extract CLS token (first token) for classification
        cls_token_output = x[:, 0]  # (B, D)
        
        # Apply our own learnable classification head
        output = self.classifier(cls_token_output)  # (B, num_classes)
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            output = output.squeeze(0)
        
        return output


def create_splattovit(
    latent_seq_len: int = 196,
    embed_dim: int = 768,
    vit_model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    **kwargs
) -> SplatToViT:
    """
    Factory function to create a SplatToViT model.
    
    Args:
        latent_seq_len: Number of latent tokens (default: 196).
        embed_dim: Embedding dimension (default: 768).
        vit_model_name: Name of ViT model from timm (default: "vit_base_patch16_224").
        pretrained: Whether to use pretrained ViT weights (default: True).
        num_classes: Number of output classes (default: 1000).
        **kwargs: Additional arguments passed to SplatToViT.
    
    Returns:
        SplatToViT instance
    """
    return SplatToViT(
        latent_seq_len=latent_seq_len,
        embed_dim=embed_dim,
        vit_model_name=vit_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )

