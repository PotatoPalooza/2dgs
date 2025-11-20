#!/usr/bin/env python3
"""
Script to download ImageNet-1k dataset from HuggingFace and save locally.
Usage:
    python scripts/download_imagenet.py --save_dir ./data/imagenet --split validation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imagenet_dataloader import download_imagenet


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet-1k dataset")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/imagenet",
        help="Directory to save the downloaded images"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use cached login if not provided)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to download (None for full dataset). Useful for testing."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume - re-download existing images"
    )
    
    args = parser.parse_args()
    
    print(f"Downloading ImageNet-1k {args.split} set...")
    print(f"Save directory: {args.save_dir}")
    print("Note: This requires HuggingFace login. Run 'huggingface-cli login' first.")
    print()
    
    try:
        download_imagenet(
            save_dir=args.save_dir,
            split=args.split,
            hf_token=args.hf_token,
            max_images=args.max_images,
            resume=not args.no_resume
        )
        if args.max_images:
            print(f"\n✓ Successfully downloaded {args.max_images} images from ImageNet-1k {args.split} set to {args.save_dir}")
        else:
            print(f"\n✓ Successfully downloaded ImageNet-1k {args.split} set to {args.save_dir}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

