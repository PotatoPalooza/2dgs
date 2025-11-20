#!/usr/bin/env python3
"""
Example script showing how to use the ImageNet dataloader.
Usage:
    python scripts/example_imagenet_usage.py --data_dir ./data/imagenet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imagenet_dataloader import get_imagenet_data_loader, download_imagenet


def main():
    parser = argparse.ArgumentParser(description="Example ImageNet dataloader usage")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/imagenet",
        help="Path to ImageNet dataset root directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--download_images",
        type=int,
        default=10,
        help="Number of images to download if dataset doesn't exist (0 to skip download)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading images even if dataset doesn't exist"
    )
    
    args = parser.parse_args()
    
    data_dir_path = Path(args.data_dir)
    split_dir = data_dir_path / args.split
    
    # Check if dataset exists, if not download a small sample
    if not split_dir.exists() or len(list(split_dir.rglob("*.JPEG"))) + len(list(split_dir.rglob("*.jpg"))) == 0:
        if args.skip_download:
            print(f"Error: Dataset not found at {split_dir}")
            print("Please download ImageNet first or run without --skip_download")
            sys.exit(1)
        
        if args.download_images > 0:
            print(f"Dataset not found at {args.data_dir}. Downloading {args.download_images} images...")
            print("Note: This requires HuggingFace login. Run 'huggingface-cli login' first if needed.")
            print()
            try:
                download_imagenet(
                    save_dir=args.data_dir,
                    split=args.split,
                    max_images=args.download_images,
                    resume=True
                )
                print()
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Please ensure you are logged in via `huggingface-cli login` and have accepted the terms.")
                sys.exit(1)
        else:
            print(f"Error: Dataset not found at {split_dir}")
            print("Please download ImageNet first or set --download_images > 0")
            sys.exit(1)
    
    print(f"Creating ImageNet dataloader for {args.split} split...")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Create dataloader
    dataloader = get_imagenet_data_loader(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        split=args.split,
        shuffle=False
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print()
    
    # Iterate through a few batches
    print("Processing samples...")
    for i, (images, gaussians) in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        if gaussians is not None:
            print(f"  Gaussians loaded: {len(gaussians)}")
        else:
            print(f"  Gaussians: None (not loading saved splats)")
        print()
    
    print("✓ Example completed successfully!")


if __name__ == "__main__":
    main()

