#!/usr/bin/env python3
"""
Load a trained renderer model and render images from splat files.

This script loads a trained GaussianRenderer from a checkpoint and uses it to render
images from Gaussian splat .pt files.

Usage:
    # Render a single splat file
    python scripts/render_splats.py \
        --checkpoint ./output/renderer/checkpoint_best.pt \
        --splat_file ./data/imagenet_splats/train/0/img1.pt \
        --output ./rendered_image.png
    
    # Render all splats in a directory
    python scripts/render_splats.py \
        --checkpoint ./output/renderer/checkpoint_best.pt \
        --splat_dir ./data/imagenet_splats/validation/0 \
        --output_dir ./rendered_images
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.renderer import GaussianRenderer


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GaussianRenderer:
    """
    Load a GaussianRenderer model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        Loaded GaussianRenderer model in eval mode
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model arguments from checkpoint
    args = checkpoint.get('args', {})
    
    # Extract model parameters
    latent_seq_len = args.get('latent_seq_len', 196)
    embed_dim = args.get('embed_dim', 768)
    num_heads = args.get('num_heads', 8)
    patch_size = args.get('patch_size', 16)
    image_size = args.get('image_size', 224)
    
    print(f"  Model config:")
    print(f"    Latent seq len: {latent_seq_len}")
    print(f"    Embed dim: {embed_dim}")
    print(f"    Num heads: {num_heads}")
    print(f"    Patch size: {patch_size}")
    print(f"    Image size: {image_size}")
    
    # Create model
    model = GaussianRenderer(
        latent_seq_len=latent_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        image_size=image_size,
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully!")
    
    return model


def load_splat_file(splat_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a splat file and return splat data dictionary.
    
    Args:
        splat_path: Path to the .pt splat file
    
    Returns:
        Dictionary with splat components: 'xy', 'scaling', 'rotation', 'color'
    """
    if not splat_path.exists():
        raise FileNotFoundError(f"Splat file not found: {splat_path}")
    
    splat_data = torch.load(splat_path, map_location='cpu')
    
    # Extract required fields
    splat_dict = {
        'xy': splat_data['xy'],  # (M, 2)
        'scaling': splat_data['scaling'],  # (M, 2)
        'rotation': splat_data['rotation'],  # (M, 1)
        'color': splat_data['color'],  # (M, 3)
    }
    
    return splat_dict


def tensor_to_image(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image to PIL Image.
    
    Args:
        image_tensor: Tensor of shape (3, H, W) in range [0, 1]
    
    Returns:
        PIL Image
    """
    # Clamp to [0, 1] and convert to [0, 255]
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_tensor = (image_tensor * 255).byte()
    
    # Convert to numpy and then PIL Image
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    image = Image.fromarray(image_np, mode='RGB')
    
    return image


def render_single_splat(
    model: GaussianRenderer,
    splat_data: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Render a single splat to an image.
    
    Args:
        model: GaussianRenderer model
        splat_data: Dictionary with splat components
        device: Device to run inference on
    
    Returns:
        Tensor of shape (3, H, W) - rendered image
    """
    # Move splat data to device
    splat_data_device = {k: v.to(device) for k, v in splat_data.items()}
    
    # Create attention mask (all True since we're using all splats)
    num_splats = splat_data_device['xy'].shape[0]
    attention_mask = torch.ones(num_splats, dtype=torch.bool, device=device)
    
    # Render
    with torch.no_grad():
        rendered_image = model(splat_data_device, attention_mask=attention_mask)
    
    return rendered_image


def main():
    parser = argparse.ArgumentParser(
        description="Render images from splat files using a trained renderer"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file containing trained renderer model"
    )
    parser.add_argument(
        "--splat_file",
        type=str,
        default=None,
        help="Path to a single splat .pt file to render"
    )
    parser.add_argument(
        "--splat_dir",
        type=str,
        default=None,
        help="Path to directory containing splat .pt files to render"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single image (used with --splat_file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for rendered images (used with --splat_dir)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available"
    )
    parser.add_argument(
        "--max_splats",
        type=int,
        default=None,
        help="Optional maximum number of splats to use per image. If not specified, use all splats."
    )
    parser.add_argument(
        "--limit_method",
        type=str,
        default="random",
        choices=["random", "opacity", "importance"],
        help="Method to use when limiting splats: 'random' (random selection), 'opacity' (by scaling/opacity), or 'importance' (by scaling * color intensity). Default: 'random'"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.splat_file is None and args.splat_dir is None:
        parser.error("Must specify either --splat_file or --splat_dir")
    if args.splat_file is not None and args.splat_dir is not None:
        parser.error("Cannot specify both --splat_file and --splat_dir")
    if args.splat_file is not None and args.output is None:
        parser.error("Must specify --output when using --splat_file")
    if args.splat_dir is not None and args.output_dir is None:
        parser.error("Must specify --output_dir when using --splat_dir")
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(Path(args.checkpoint), device)
    
    # Process single file
    if args.splat_file is not None:
        splat_path = Path(args.splat_file)
        output_path = Path(args.output)
        
        print(f"\nRendering {splat_path}...")
        splat_data = load_splat_file(splat_path)
        
        # Apply splat limit if specified
        if args.max_splats is not None:
            num_splats = splat_data['xy'].shape[0]
            if num_splats > args.max_splats:
                if args.limit_method == "random":
                    indices = torch.randperm(num_splats)[:args.max_splats]
                elif args.limit_method == "opacity":
                    avg_scaling = splat_data['scaling'].mean(dim=1)
                    _, indices = torch.topk(avg_scaling, args.max_splats)
                elif args.limit_method == "importance":
                    avg_scaling = splat_data['scaling'].mean(dim=1)
                    color_intensity = splat_data['color'].mean(dim=1)
                    importance = avg_scaling * color_intensity
                    _, indices = torch.topk(importance, args.max_splats)
                
                for key in splat_data:
                    splat_data[key] = splat_data[key][indices]
                print(f"  Limited to {args.max_splats} splats (method: {args.limit_method})")
        
        # Render
        rendered_image = render_single_splat(model, splat_data, device)
        
        # Convert to PIL Image and save
        image = tensor_to_image(rendered_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"  Saved to {output_path}")
    
    # Process directory
    else:
        splat_dir = Path(args.splat_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .pt files
        splat_files = sorted(splat_dir.glob("*.pt"))
        
        if len(splat_files) == 0:
            print(f"No .pt files found in {splat_dir}")
            return
        
        print(f"\nFound {len(splat_files)} splat files in {splat_dir}")
        print(f"Rendering to {output_dir}...")
        
        for splat_path in tqdm(splat_files, desc="Rendering"):
            try:
                splat_data = load_splat_file(splat_path)
                
                # Apply splat limit if specified
                if args.max_splats is not None:
                    num_splats = splat_data['xy'].shape[0]
                    if num_splats > args.max_splats:
                        if args.limit_method == "random":
                            indices = torch.randperm(num_splats)[:args.max_splats]
                        elif args.limit_method == "opacity":
                            avg_scaling = splat_data['scaling'].mean(dim=1)
                            _, indices = torch.topk(avg_scaling, args.max_splats)
                        elif args.limit_method == "importance":
                            avg_scaling = splat_data['scaling'].mean(dim=1)
                            color_intensity = splat_data['color'].mean(dim=1)
                            importance = avg_scaling * color_intensity
                            _, indices = torch.topk(importance, args.max_splats)
                        
                        for key in splat_data:
                            splat_data[key] = splat_data[key][indices]
                
                # Render
                rendered_image = render_single_splat(model, splat_data, device)
                
                # Convert to PIL Image and save
                image = tensor_to_image(rendered_image)
                output_path = output_dir / f"{splat_path.stem}.png"
                image.save(output_path)
                
            except Exception as e:
                print(f"\nError processing {splat_path}: {e}")
                continue
        
        print(f"\n✓ Rendered {len(splat_files)} images to {output_dir}")


if __name__ == "__main__":
    main()



