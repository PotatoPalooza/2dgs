"""
Adapter model that converts Gaussian splats to ViT-compatible features.

The adapter takes in splat data and produces 768-dimensional features per splat:
- Fourier analysis of x,y coordinates -> R^24 per splat
- MLP processing of scaling, rotation, color -> R^744 per splat
- Concatenated output -> R^768 per splat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module where latent vectors attend to splat features.

    - Queries: latent vectors (B, L, D)
    - Keys/Values: splat features (B, M, D)
    - attention_mask: (B, M) masks splats (keys/values)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        L: int = 196,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Embedding dimension (default: 768 for ViT).
            num_heads: Number of attention heads.
            L: Length of latent sequence (number of latent tokens). Used mainly as a config hint.
            dropout: Dropout probability.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.L = L
        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        latent_vectors: torch.Tensor,
        splat_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent_vectors: (B, L, D) - latent tokens (queries)
            splat_features: (B, M, D) - splat features (keys/values)
            attention_mask: (B, M) - 1/True for real splat, 0/False for padding

        Returns:
            (B, L, D) - updated latent vectors
        """
        B, L, D = latent_vectors.shape
        _, M, Dk = splat_features.shape
        assert D == self.embed_dim and Dk == self.embed_dim

        # Residual on latents
        residual = latent_vectors

        # Linear projections
        q = self.q_proj(latent_vectors)   # (B, L, D)
        k = self.k_proj(splat_features)   # (B, M, D)
        v = self.v_proj(splat_features)   # (B, M, D)

        # Reshape to multi-head: (B, H, L, head_dim) and (B, H, M, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, Hd)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, Hd)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, Hd)

        # Attention scores: (B, H, L, M)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask over splats (keys/values)
        if attention_mask is not None:
            # attention_mask: (B, M) with 1/True = real, 0/False = pad
            if attention_mask.dtype == torch.bool:
                pad_mask = ~attention_mask  # True where padding
            else:
                pad_mask = (attention_mask == 0)

            # Convert to logits mask: 0 for real, -1e9 for padding
            mask_logits = pad_mask.float() * -1e9  # (B, M)
            # Expand to (B, 1, 1, M) so it broadcasts over heads and L
            mask_logits = mask_logits.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, M)
            attn_scores = attn_scores + mask_logits

        # Attention weights: (B, H, L, M)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: (B, H, L, Hd)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (B, L, H, Hd) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Residual + layer norm on latents
        output = self.layer_norm(output + residual)  # (B, L, D)

        return output


class GaussianAdapter(nn.Module):
    """
    Adapter that converts Gaussian splat data to 768-dimensional ViT features per splat.
    
    The adapter consists of two branches:
    1. Fourier branch: Processes x,y coordinates using Fourier analysis -> R^24 per splat
    2. MLP branch: Processes scaling, rotation, color -> R^744 per splat
    The outputs are concatenated to produce R^768 features per splat.
    
    Optionally includes a cross-attention module where latent vectors (queries) attend to 
    splat features (keys/values). This allows handling variable numbers of gaussian splats 
    per image and updates the latent vectors based on the splat information.
    The cross-attention returns updated latent vectors (196 tokens, same as ViT).
    
    Supports both batched and unbatched inputs.
    """
    
    def __init__(
        self,
        fourier_freqs: int = 6,
        mlp_hidden_dims: Optional[list] = None,
        output_dim: int = 768,
        fourier_dim: int = 24,
        num_heads: int = 8,
        latent_seq_len: int = 196,
        cross_attn_dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        """
        Initialize the adapter model.
        
        Args:
            fourier_freqs: Number of frequencies to use in Fourier analysis.
                          With 6 frequencies, we get 2*2*6=24 features (sin/cos for x and y).
            mlp_hidden_dims: Hidden dimensions for the MLP. If None, uses [256, 512, 744].
            output_dim: Total output dimension (default: 768 for ViT).
            fourier_dim: Dimension of Fourier features (default: 24).
            num_heads: Number of attention heads for cross-attention (default: 8).
            latent_seq_len: Length of latent sequence for cross-attention (default: 196 for ViT tokens).
            cross_attn_dropout: Dropout probability for cross-attention (default: 0.1).
            use_cross_attention: Whether to use cross-attention module (default: True).
        """
        super().__init__()
        
        self.fourier_freqs = fourier_freqs
        self.fourier_dim = fourier_dim
        self.output_dim = output_dim
        self.mlp_input_dim = 6  # scaling (2) + rotation (1) + color (3)
        self.mlp_output_dim = output_dim - fourier_dim  # 768 - 24 = 744
        self.use_cross_attention = use_cross_attention
        
        # Verify Fourier dimension matches number of frequencies
        expected_fourier_dim = 2 * 2 * fourier_freqs  # sin/cos for x and y
        if fourier_dim != expected_fourier_dim:
            raise ValueError(
                f"fourier_dim ({fourier_dim}) must equal 2*2*fourier_freqs "
                f"({expected_fourier_dim}) for sin/cos encoding of x and y"
            )
        
        # Define frequencies for Fourier encoding
        # Using powers of 2: 2^0, 2^1, ..., 2^(fourier_freqs-1)
        self.register_buffer(
            'fourier_frequencies',
            torch.tensor([2.0 ** i for i in range(fourier_freqs)], dtype=torch.float32)
        )
        
        # MLP for processing scaling, rotation, and color
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 512, self.mlp_output_dim]
        
        mlp_layers = []
        input_dim = self.mlp_input_dim
        for hidden_dim in mlp_hidden_dims[:-1]:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim
        
        # Final layer to output dimension
        mlp_layers.append(nn.Linear(input_dim, mlp_hidden_dims[-1]))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Cross-attention module
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                L=latent_seq_len,
                dropout=cross_attn_dropout,
            )
        else:
            self.cross_attention = None
        
    def fourier_encode_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Perform Fourier encoding on x,y coordinates.
        
        Args:
            xy: Tensor of shape [B, M, 2] containing x,y coordinates (must be batched)
        
        Returns:
            Tensor of shape [B, M, fourier_dim] with Fourier-encoded features
        """
        # xy: [B, M, 2]
        x = xy[:, :, 0:1]  # [B, M, 1]
        y = xy[:, :, 1:2]  # [B, M, 1]
        
        # Expand frequencies: [fourier_freqs] -> [1, 1, fourier_freqs] for broadcasting
        freqs = self.fourier_frequencies.unsqueeze(0).unsqueeze(0)  # [1, 1, fourier_freqs]
        
        # Compute sin and cos for x and y at each frequency: [B, M, fourier_freqs]
        x_sin = torch.sin(x * freqs * math.pi)
        x_cos = torch.cos(x * freqs * math.pi)
        y_sin = torch.sin(y * freqs * math.pi)
        y_cos = torch.cos(y * freqs * math.pi)
        
        # Concatenate along feature dimension
        fourier_features = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)  # [B, M, fourier_dim]
        
        return fourier_features
    
    def forward(
        self,
        splat_data: Dict[str, torch.Tensor],
        latent_vectors: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            splat_data: Dictionary containing splat components (batched or unbatched):
                - 'xy': Tensor of shape [B, M, 2] or [M, 2] - x,y coordinates
                - 'scaling': Tensor of shape [B, M, 2] or [M, 2] - scaling parameters
                - 'rotation': Tensor of shape [B, M, 1] or [M, 1] - rotation parameter
                - 'color': Tensor of shape [B, M, 3] or [M, 3] - color values
            latent_vectors: Optional tensor of shape [B, L, output_dim] or [L, output_dim] 
                          - latent tokens (required if use_cross_attention=True).
                          L is typically 196 for ViT tokens.
            attention_mask: Optional tensor of shape [B, M] or [M] - 1/True if real splat, 
                          0/False if padding (boolean or integer tensor, used when 
                          use_cross_attention=True).
        
        Returns:
            If use_cross_attention=True:
                Tensor of shape [B, L, output_dim] or [L, output_dim] - updated latent vectors
            If use_cross_attention=False:
                Tensor of shape [B, M, output_dim] or [M, output_dim] - splat features
        """
        xy = splat_data['xy']
        scaling = splat_data['scaling']
        rotation = splat_data['rotation']
        color = splat_data['color']
        
        # Add batch dimension if missing
        was_unbatched = xy.dim() == 2
        if was_unbatched:
            xy = xy.unsqueeze(0)  # [M, 2] -> [1, M, 2]
            scaling = scaling.unsqueeze(0)  # [M, 2] -> [1, M, 2]
            rotation = rotation.unsqueeze(0)  # [M, 1] -> [1, M, 1]
            color = color.unsqueeze(0)  # [M, 3] -> [1, M, 3]
        
        # Now everything is batched: [B, M, ...]
        B, M = xy.shape[:2]
        
        # Flatten batch and sequence for processing
        xy_flat = xy.view(B * M, -1)  # [B*M, 2]
        scaling_flat = scaling.view(B * M, -1)  # [B*M, 2]
        rotation_flat = rotation.view(B * M, -1)  # [B*M, 1]
        color_flat = color.view(B * M, -1)  # [B*M, 3]
        
        # Reshape for fourier encoding (needs [B, M, 2])
        xy_reshaped = xy_flat.view(B, M, -1)  # [B, M, 2]
        
        # Fourier branch: process x,y coordinates per splat
        fourier_features = self.fourier_encode_xy(xy_reshaped)  # [B, M, fourier_dim]
        fourier_features_flat = fourier_features.view(B * M, -1)  # [B*M, fourier_dim]
        
        # MLP branch: process scaling, rotation, and color per splat
        mlp_input = torch.cat([scaling_flat, rotation_flat, color_flat], dim=-1)  # [B*M, 6]
        mlp_features = self.mlp(mlp_input)  # [B*M, mlp_output_dim]
        
        # Concatenate Fourier and MLP features per splat
        splat_features_flat = torch.cat([fourier_features_flat, mlp_features], dim=-1)  # [B*M, output_dim]
        splat_features = splat_features_flat.view(B, M, self.output_dim)  # [B, M, output_dim]
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            if latent_vectors is None:
                raise ValueError(
                    "latent_vectors must be provided when use_cross_attention=True"
                )
            
            # Add batch dimension to latent_vectors if missing
            if latent_vectors.dim() == 2:
                latent_vectors = latent_vectors.unsqueeze(0).expand(B, -1, -1)  # [L, D] -> [1, L, D] -> [B, L, D]
            elif latent_vectors.shape[0] != B:
                raise ValueError(
                    f"Mismatch: splat_data batch size {B} != latent_vectors batch size {latent_vectors.shape[0]}"
                )
            
            # Add batch dimension to attention_mask if missing
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0).expand(B, -1)  # [M] -> [1, M] -> [B, M]
                elif attention_mask.shape[0] != B:
                    raise ValueError(
                        f"Mismatch: splat_data batch size {B} != attention_mask batch size {attention_mask.shape[0]}"
                    )
            
            # Cross-attention: latent vectors attend to splat features
            output = self.cross_attention(
                latent_vectors=latent_vectors,
                splat_features=splat_features,
                attention_mask=attention_mask,
            )  # [B, L, output_dim]
            
            # Remove batch dimension if input was unbatched
            if was_unbatched:
                output = output.squeeze(0)  # [1, L, output_dim] -> [L, output_dim]
            
            return output
        else:
            # Remove batch dimension if input was unbatched
            if was_unbatched:
                splat_features = splat_features.squeeze(0)  # [1, M, output_dim] -> [M, output_dim]
            return splat_features


def create_adapter(
    fourier_freqs: int = 6,
    output_dim: int = 768,
    **kwargs
) -> GaussianAdapter:
    """
    Factory function to create a GaussianAdapter.
    
    Args:
        fourier_freqs: Number of frequencies for Fourier encoding (default: 6)
        output_dim: Output dimension (default: 768)
        **kwargs: Additional arguments passed to GaussianAdapter
    
    Returns:
        GaussianAdapter instance
    """
    return GaussianAdapter(
        fourier_freqs=fourier_freqs,
        output_dim=output_dim,
        **kwargs
    )

