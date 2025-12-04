#!/usr/bin/env python3
"""
Compress ImageNet images using Instant-GI (2D Gaussian Splatting) and rasterize them back to images.

This script can operate in two modes:
1. Compress mode: Loads images, compresses them using Instant-GI model, and rasterizes them
2. Rasterize mode: Loads pre-generated splats (.pt files) and rasterizes them to images

Usage (Compress mode):
    python scripts/compress_imagenet_instantgi.py \
        --input_dir ./data/imagenet/validation \
        --output_dir ./output/compressed_imagenet \
        --checkpoint ./Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth

Usage (Rasterize mode):
    python scripts/compress_imagenet_instantgi.py \
        --input_dir ./output/imagenet_gaussians \
        --output_dir ./output/compressed_imagenet \
        --mode rasterize \
        --image_size 224 224 \
        --max_splats 10000 \
        --limit_method importance
"""

import argparse
import sys
from pathlib import Path
import os
from typing import Optional
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add Instant-GI to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "Instant-GI"))

from generalizable_model.init_net import InitNet, render


def limit_splats(gaussians: dict, max_splats: int, method: str = "importance"):
    """
    Limit the number of splats in the gaussians dictionary.
    
    Args:
        gaussians: Dictionary with keys 'xy', 'scaling', 'rotation', 'color', 'triangles'
        max_splats: Maximum number of splats to keep
        method: Method to select splats - 'random', 'opacity' (by scaling), or 'importance'
    
    Returns:
        Filtered gaussians dictionary
    """
    num_splats = gaussians['xy'].shape[0]
    if num_splats <= max_splats:
        return gaussians
    
    if method == "random":
        indices = torch.randperm(num_splats, device=gaussians['xy'].device)[:max_splats]
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
    return result


def load_splats_from_file(splat_path: Path, device: torch.device) -> dict:
    """
    Load Gaussian splats from a .pt file.
    
    Args:
        splat_path: Path to the .pt file containing splats
        device: Device to load tensors on
    
    Returns:
        Dictionary containing Gaussian parameters: xy, scaling, rotation, color, triangles
    """
    if not splat_path.exists():
        raise FileNotFoundError(f"Splat file not found: {splat_path}")
    
    gaussians = torch.load(splat_path, map_location=device)
    
    # Ensure all tensors are on the correct device
    return {
        "xy": gaussians["xy"].to(device),
        "scaling": gaussians["scaling"].to(device),
        "rotation": gaussians["rotation"].to(device),
        "color": gaussians["color"].to(device),
        "triangles": gaussians.get("triangles", None),
    }


def setup_model(checkpoint_path: str, device: torch.device):
    """
    Load and setup the Instant-GI model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
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


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        device: Device to load the tensor on
    
    Returns:
        Preprocessed image tensor [1, 3, H, W] in range [0, 1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor


def compress_to_gaussians(model: InitNet, image_tensor: torch.Tensor):
    """
    Compress an image into Gaussians using Instant-GI.
    
    Args:
        model: Instant-GI model
        image_tensor: Image tensor [1, 3, H, W]
    
    Returns:
        Dictionary containing Gaussian parameters: xy, scaling, rotation, color, triangles
    """
    with torch.no_grad():
        xy, scaling, rotation, color, triangles = model(image_tensor, get_gaussians=True)
    
    return {
        "xy": xy.squeeze(0),
        "scaling": scaling.squeeze(0),
        "rotation": rotation.squeeze(0),
        "color": color.squeeze(0),
        "triangles": triangles,
    }


def rasterize_gaussians(xy, scaling, rotation, color, H, W, device: torch.device) -> torch.Tensor:
    """
    Rasterize Gaussians back into an image.
    
    Args:
        xy: Gaussian positions [N, 2]
        scaling: Gaussian scales [N, 2]
        rotation: Gaussian rotations [N, 1] (in range [0, 1], will be converted to [0, 2π])
        color: Gaussian colors [N, 3]
        H: Image height
        W: Image width
        device: Device to perform computation on
    
    Returns:
        Rendered image tensor [1, 3, H, W] in range [0, 1]
    """
    # Convert rotation from [0, 1] to [0, 2π] as done in the model forward
    rotation = rotation * 2 * torch.pi
    
    # Use the render function from init_net
    result = render(xy, scaling, rotation, color, H, W)
    rendered_image = result["render"]  # [1, 3, H, W]
    
    # Clamp to [0, 1] range
    rendered_image = torch.clamp(rendered_image, 0, 1)
    
    return rendered_image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image to PIL Image.
    
    Args:
        image_tensor: Image tensor [1, 3, H, W] or [3, H, W] in range [0, 1]
    
    Returns:
        PIL Image
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # Convert from [C, H, W] to [H, W, C] and to numpy
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert from [0, 1] to [0, 255]
    image_np = (image_np * 255).clip(0, 255).astype('uint8')
    
    return Image.fromarray(image_np)


def process_image(
    image_path: Path,
    model: Optional[InitNet],
    output_path: Path,
    device: torch.device,
    mode: str = "compress",
    image_size: Optional[tuple] = None,
    max_splats: Optional[int] = None,
    limit_method: str = "importance"
):
    """
    Process a single image: load, compress (if needed), rasterize, and save.
    
    Args:
        image_path: Path to input image (or .pt file in rasterize mode)
        model: Instant-GI model (can be None in rasterize mode)
        output_path: Path to save rendered image
        device: Device to perform computation on
        mode: 'compress' or 'rasterize'
        image_size: (H, W) tuple for rasterize mode. If None, will try to infer from original image
        max_splats: Optional maximum number of splats to use (rasterize mode only)
        limit_method: Method to use when limiting splats ('random', 'opacity', or 'importance')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if mode == "compress":
            # Load image
            image_tensor = load_image(image_path, device)
            _, _, H, W = image_tensor.shape
            
            # Compress to Gaussians
            gaussians = compress_to_gaussians(model, image_tensor)
            
        elif mode == "rasterize":
            # Load splats from .pt file
            gaussians = load_splats_from_file(image_path, device)
            
            # Limit splats if requested
            if max_splats is not None:
                gaussians = limit_splats(gaussians, max_splats, method=limit_method)
            
            # Determine image size
            if image_size is not None:
                H, W = image_size
            else:
                # Try to infer from the original image path
                # Assuming .pt files correspond to images with same name
                # This is a fallback - image_size should be provided
                # For now, use a default or try to load original image
                raise ValueError("image_size must be provided in rasterize mode")
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Rasterize back to image
        rendered_image = rasterize_gaussians(
            gaussians["xy"],
            gaussians["scaling"],
            gaussians["rotation"],
            gaussians["color"],
            H, W,
            device
        )
        
        # Save rendered image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image = tensor_to_pil(rendered_image)
        pil_image.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    model: Optional[InitNet],
    device: torch.device,
    mode: str = "compress",
    image_extensions: Optional[set] = None,
    image_size: Optional[tuple] = None,
    max_splats: Optional[int] = None,
    limit_method: str = "importance"
):
    """
    Process all images/splats in a directory recursively.
    
    Args:
        input_dir: Input directory containing images or .pt files
        output_dir: Output directory to save rendered images
        model: Instant-GI model (can be None in rasterize mode)
        device: Device to perform computation on
        mode: 'compress' or 'rasterize'
        image_extensions: Set of image file extensions to process (for compress mode)
        image_size: (H, W) tuple for rasterize mode
        max_splats: Optional maximum number of splats per image (rasterize mode only)
        limit_method: Method to use when limiting splats ('random', 'opacity', or 'importance')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all files recursively
    if mode == "compress":
        # Find image files
        if image_extensions is None:
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.PNG'}
        files = []
        for ext in image_extensions:
            files.extend(input_dir.rglob(f'*{ext}'))
        file_type = "images"
    elif mode == "rasterize":
        # Find .pt files
        files = list(input_dir.rglob('*.pt'))
        file_type = "splat files"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if len(files) == 0:
        print(f"No {file_type} found in {input_dir}")
        return
    
    print(f"Found {len(files)} {file_type} to process")
    
    # Process each file
    success_count = 0
    for file_path in tqdm(files, desc=f"Processing {file_type}"):
        # Maintain directory structure in output
        relative_path = file_path.relative_to(input_dir)
        
        # Change extension to .png for output
        if mode == "rasterize":
            # .pt -> .png
            output_path = output_dir / relative_path.with_suffix('.png')
        else:
            # Keep original extension or convert to .png
            output_path = output_dir / relative_path.with_suffix('.png')
        
        if process_image(file_path, model, output_path, device, mode=mode, image_size=image_size, 
                        max_splats=max_splats, limit_method=limit_method):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(files)} {file_type} successfully")
    print(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compress ImageNet images using Instant-GI and rasterize them back, "
                    "or rasterize pre-generated splats"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing images (compress mode) or .pt files (rasterize mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save rasterized images"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compress",
        choices=["compress", "rasterize"],
        help="Mode: 'compress' (use Instant-GI model) or 'rasterize' (load from .pt files). Default: compress"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Instant-GI checkpoint file (required for compress mode). "
             f"Defaults to {REPO_ROOT}/Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Image size (height width) for rasterize mode. Required when mode=rasterize"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.JPEG,.PNG",
        help="Comma-separated list of image extensions to process (compress mode only). "
             "Default: .jpg,.jpeg,.png,.JPEG,.PNG"
    )
    parser.add_argument(
        "--max_splats",
        type=int,
        default=None,
        help="Maximum number of splats per image to use (rasterize mode only). "
             "If not specified, all splats are used."
    )
    parser.add_argument(
        "--limit_method",
        type=str,
        default="importance",
        choices=["random", "opacity", "importance"],
        help="Method to use when limiting splats (rasterize mode only): "
             "'random' (random selection), 'opacity' (by scaling/opacity), "
             "or 'importance' (by scaling * color intensity). Default: 'importance'"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    
    # Load model only if in compress mode
    model = None
    if args.mode == "compress":
        # Set default checkpoint path if not provided
        if args.checkpoint is None:
            checkpoint_path = REPO_ROOT / "Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth"
        else:
            checkpoint_path = Path(args.checkpoint)
        
        print(f"Checkpoint path: {checkpoint_path}")
        print()
        
        # Load model
        try:
            model = setup_model(checkpoint_path, device)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        print()
    elif args.mode == "rasterize":
        if args.image_size is None:
            print("Error: --image_size is required when mode=rasterize")
            print("Example: --image_size 224 224")
            sys.exit(1)
        print(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
        if args.max_splats is not None:
            print(f"Max splats per image: {args.max_splats}")
            print(f"Limit method: {args.limit_method}")
        else:
            print(f"Max splats per image: No limit (using all splats)")
        print()
    
    # Process files
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Parse extensions for compress mode
    if args.mode == "compress":
        exts = set(ext.strip() for ext in args.extensions.split(','))
    else:
        exts = None
    
    print()
    
    # Convert image_size to tuple if provided
    image_size = tuple(args.image_size) if args.image_size else None
    
    process_directory(
        input_dir, output_dir, model, device,
        mode=args.mode,
        image_extensions=exts,
        image_size=image_size,
        max_splats=args.max_splats,
        limit_method=args.limit_method
    )
    
    print("\n✓ Processing completed!")


if __name__ == "__main__":
    main()

