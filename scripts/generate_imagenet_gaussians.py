#!/usr/bin/env python3
"""
Generate Gaussian splats from ImageNet images using Instant-GI.

This script processes locally stored ImageNet images and generates 2D Gaussian splats,
saving them as .pt files while preserving the ImageNet directory structure.

Usage:
    python scripts/generate_imagenet_gaussians.py \
        --input_dir ./data/imagenet \
        --output_dir ./output/imagenet_gaussians \
        --checkpoint ./Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth \
        --max_splats 10000 \
        --limit_method importance
"""

import argparse
import sys
from pathlib import Path
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "Instant-GI"))

from generalizable_model.init_net import InitNet


def limit_splats(gaussians: dict, max_splats: int, method: str = "importance"):
    """
    Limit the number of splats in the gaussians dictionary.
    
    Args:
        gaussians: Dictionary with keys 'xy', 'scaling', 'rotation', 'color', 'triangles'
        max_splats: Maximum number of splats to keep
        method: Method to select splats - 'random', 'opacity' (by scaling), or 'importance'
    
    Returns:
        Tuple of (filtered gaussians dictionary, original count)
    """
    num_splats = gaussians['xy'].shape[0]
    original_count = num_splats
    if num_splats <= max_splats:
        return gaussians, original_count
    
    if method == "random":
        indices = torch.randperm(num_splats)[:max_splats]
    elif method == "opacity":
        # Sort by average scaling (larger = more important)
        avg_scaling = gaussians['scaling'].mean(dim=1)
        _, indices = torch.topk(avg_scaling, max_splats)
    elif method == "importance":
        # Combine scaling and color intensity as importance metric
        avg_scaling = gaussians['scaling'].mean(dim=1)
        color_intensity = gaussians['color'].mean(dim=1)
        importance = avg_scaling * color_intensity
        _, indices = torch.topk(importance, max_splats)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'random', 'opacity', or 'importance'")
    
    # Filter all tensors
    result = {
        'xy': gaussians['xy'][indices],
        'scaling': gaussians['scaling'][indices],
        'rotation': gaussians['rotation'][indices],
        'color': gaussians['color'][indices],
        'triangles': gaussians['triangles'][indices] if gaussians['triangles'] is not None else None,
    }
    return result, original_count


def setup_model(checkpoint_path: Path, device: torch.device):
    """
    Load and setup the Instant-GI model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        Loaded model in eval mode
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please download the checkpoint or specify a different path."
        )
    
    print(f"Loading model from {checkpoint_path}...")
    model = InitNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Model loaded successfully!")
    return model


def preprocess(image_path: Path) -> torch.Tensor:
    """
    Preprocess an image file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor [3, H, W] in range [0, 1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(Image.open(image_path).convert("RGB"))


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        device: Device to load the tensor on
    
    Returns:
        Preprocessed image tensor [1, 3, H, W] in range [0, 1]
    """
    image_tensor = preprocess(image_path).unsqueeze(0).to(device)
    return image_tensor


def infer(model: InitNet, img_t: torch.Tensor, device: torch.device, max_splats: int = None, limit_method: str = "importance"):
    """
    Run inference to generate Gaussian splats from an image.
    
    Args:
        model: Instant-GI model
        img_t: Image tensor [1, 3, H, W] or [3, H, W]
        device: Device to perform computation on
        max_splats: Optional maximum number of splats to return
        limit_method: Method to use when limiting splats ('random', 'opacity', or 'importance')
    
    Returns:
        Tuple of (gaussians dictionary, original_count). If max_splats is None, original_count equals final count.
    """
    # Ensure batch dimension
    if img_t.dim() == 3:
        img_t = img_t.unsqueeze(0)
    image_tensor = img_t.to(device)
    
    with torch.no_grad():
        xy, scaling, rotation, color, triangles = model(image_tensor, get_gaussians=True)
    
    # Move all tensors to CPU for saving (triangles may stay on CUDA by default)
    gaussians = dict(
        xy=xy.squeeze(0).cpu(),
        scaling=scaling.squeeze(0).cpu(),
        rotation=rotation.squeeze(0).cpu(),
        color=color.squeeze(0).cpu(),
        triangles=triangles.cpu() if triangles is not None else None,
    )
    
    # Limit splats if requested
    original_count = gaussians['xy'].shape[0]
    if max_splats is not None:
        gaussians, original_count = limit_splats(gaussians, max_splats, method=limit_method)
    
    return gaussians, original_count


def process_recursive(
    src: Path,
    dst: Path,
    model: InitNet,
    device: torch.device,
    exts: set = {'.jpg', '.png', '.jpeg', '.JPEG'},
    resume: bool = True,
    max_splats: int = None,
    limit_method: str = "importance"
):
    """
    Recursively process all images in a directory and save Gaussian splats.
    
    Preserves the directory structure from input to output, replacing image
    extensions with .pt.
    
    Args:
        src: Source directory containing images
        dst: Destination directory for .pt files
        model: Instant-GI model
        device: Device to perform computation on
        exts: Set of image file extensions to process
        resume: If True, skip existing .pt files
        max_splats: Optional maximum number of splats per image
        limit_method: Method to use when limiting splats ('random', 'opacity', or 'importance')
    """
    src, dst = Path(src), Path(dst)
    
    # Find all matching files recursively
    files = [p for p in src.rglob('*') if p.suffix.lower() in exts or p.suffix in exts]
    
    if len(files) == 0:
        print(f"No images found in {src} with extensions {exts}")
        return
    
    print(f"Found {len(files)} images to process")
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # Track splat statistics
    splat_counts = []
    original_splat_counts = []
    
    for f in tqdm(files, desc="Processing"):
        # Mirror structure: src/A/B/img.jpg -> dst/A/B/img.pt
        out = dst / f.relative_to(src).with_suffix('.pt')
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file exists and resume is enabled
        if resume and out.exists():
            skipped_count += 1
            continue
        
        try:
            img = load_image(f, device)
            # Remove batch dimension for infer (it will add it back)
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            # Get gaussians (will be limited if max_splats is set)
            gaussians, original_count = infer(model, img, device, max_splats=max_splats, limit_method=limit_method)
            num_splats = gaussians['xy'].shape[0]
            splat_counts.append(num_splats)
            
            # Track original count if limiting is enabled
            if max_splats is not None:
                original_splat_counts.append(original_count)
            
            torch.save(gaussians, out)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\nSkipped {f.name}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}")
    
    # Print splat statistics
    if splat_counts:
        print(f"\nSplat statistics:")
        if max_splats is not None:
            print(f"  Max splats limit: {max_splats}")
            print(f"  Limit method: {limit_method}")
            if original_splat_counts:
                print(f"  Original splats (before limiting):")
                print(f"    Min: {min(original_splat_counts)}")
                print(f"    Max: {max(original_splat_counts)}")
                print(f"    Avg: {sum(original_splat_counts) / len(original_splat_counts):.2f}")
        print(f"  Final splats per image:")
        print(f"    Min: {min(splat_counts)}")
        print(f"    Max: {max(splat_counts)}")
        print(f"    Avg: {sum(splat_counts) / len(splat_counts):.2f}")
    else:
        print(f"\nSplat statistics: No images processed (all skipped or errors)")
    
    print(f"Output saved to {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gaussian splats from ImageNet images using Instant-GI"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing ImageNet images (can be nested)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save Gaussian splat .pt files"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Instant-GI checkpoint file. "
             f"Defaults to {REPO_ROOT}/Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip existing .pt files (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite existing .pt files"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.JPEG",
        help="Comma-separated list of image extensions to process (default: .jpg,.jpeg,.png,.JPEG)"
    )
    parser.add_argument(
        "--max_splats",
        type=int,
        default=None,
        help="Maximum number of splats per image. If not specified, no limit is applied."
    )
    parser.add_argument(
        "--limit_method",
        type=str,
        default="importance",
        choices=["random", "opacity", "importance"],
        help="Method to use when limiting splats: 'random' (random selection), "
             "'opacity' (by scaling/opacity), or 'importance' (by scaling * color intensity). "
             "Default: 'importance'"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set default checkpoint path if not provided
    if args.checkpoint is None:
        checkpoint_path = REPO_ROOT / "Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Parse extensions
    exts = set(ext.strip() for ext in args.extensions.split(','))
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Extensions: {exts}")
    print(f"Resume mode: {args.resume}")
    if args.max_splats is not None:
        print(f"Max splats per image: {args.max_splats}")
        print(f"Limit method: {args.limit_method}")
    else:
        print(f"Max splats per image: No limit")
    print()
    
    # Load model
    try:
        model = setup_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print()
    
    # Process images
    process_recursive(
        input_dir, output_dir, model, device, exts, args.resume,
        max_splats=args.max_splats, limit_method=args.limit_method
    )
    
    print("\n✓ Processing completed!")


if __name__ == "__main__":
    main()

