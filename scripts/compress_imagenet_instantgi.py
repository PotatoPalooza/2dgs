#!/usr/bin/env python3
"""
Compress ImageNet images using Instant-GI (2D Gaussian Splatting) and rasterize them back to images.

This script:
1. Loads images from an input directory
2. Compresses them into Gaussians using Instant-GI model
3. Rasterizes the Gaussians back into images
4. Saves the rendered images to an output directory

Usage:
    python scripts/compress_imagenet_instantgi.py \
        --input_dir ./data/imagenet/validation \
        --output_dir ./output/compressed_imagenet \
        --checkpoint ./Instant-GI/checkpoints/epoch_best_ks_3_cupy.pth
"""

import argparse
import sys
from pathlib import Path
import os
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
    model: InitNet,
    output_path: Path,
    device: torch.device
):
    """
    Process a single image: load, compress, rasterize, and save.
    
    Args:
        image_path: Path to input image
        model: Instant-GI model
        output_path: Path to save rendered image
        device: Device to perform computation on
    """
    try:
        # Load image
        image_tensor = load_image(image_path, device)
        _, _, H, W = image_tensor.shape
        
        # Compress to Gaussians
        gaussians = compress_to_gaussians(model, image_tensor)
        
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
    model: InitNet,
    device: torch.device,
    image_extensions: set = {'.jpg', '.jpeg', '.png', '.JPEG', '.PNG'}
):
    """
    Process all images in a directory recursively.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory to save rendered images
        model: Instant-GI model
        device: Device to perform computation on
        image_extensions: Set of image file extensions to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all image files recursively
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f'*{ext}'))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        # Maintain directory structure in output
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        if process_image(image_path, model, output_path, device):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(image_files)} images successfully")
    print(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compress ImageNet images using Instant-GI and rasterize them back"
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
        help="Output directory to save rasterized images"
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
    print()
    
    # Load model
    try:
        model = setup_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print()
    
    # Process images
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    process_directory(input_dir, output_dir, model, device)
    
    print("\n✓ Processing completed!")


if __name__ == "__main__":
    main()

