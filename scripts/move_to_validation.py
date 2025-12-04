#!/usr/bin/env python3
"""
Move corresponding pairs of images and splats from training to validation sets.

This script moves 10000 corresponding pairs from:
  - /workspace/data/train -> /workspace/data/validation
  - /workspace/splats/train/nolimit -> /workspace/splats/validation/nolimit

It tries to take at least 1 pair from each class (if possible) without leaving
any class empty in the training set.
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def find_paired_files(train_images_dir, train_splats_dir):
    """
    Find all paired image and splat files across all classes.
    
    Returns:
        dict: {class_label: [(image_path, splat_path), ...]}
    """
    train_images_dir = Path(train_images_dir)
    train_splats_dir = Path(train_splats_dir)
    
    pairs_by_class = defaultdict(list)
    
    # Find all class directories
    for label_dir in sorted(train_splats_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        
        try:
            label = int(label_dir.name)
        except ValueError:
            # Skip non-numeric directories
            continue
        
        # Find corresponding image directory
        image_label_dir = train_images_dir / label_dir.name
        if not image_label_dir.exists():
            print(f"Warning: Image directory {image_label_dir} does not exist, skipping class {label}")
            continue
        
        # Find all .pt files in this label directory
        for splat_path in sorted(label_dir.glob("*.pt")):
            # Find corresponding image file (try common extensions)
            image_name = splat_path.stem  # filename without .pt extension
            image_path = None
            for ext in ['.jpeg', '.jpg', '.JPEG', '.JPG', '.png', '.PNG']:
                candidate = image_label_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                print(f"Warning: No corresponding image found for {splat_path}, skipping")
                continue
            
            pairs_by_class[label].append((image_path, splat_path))
    
    return pairs_by_class


def select_pairs_to_move(pairs_by_class, target_count=10000):
    """
    Select pairs to move, trying to get at least 1 from each class.
    
    Strategy:
    1. For classes with > 1 pair, we can move at least 1 (but not all)
    2. For classes with exactly 1 pair, skip (to avoid leaving empty)
    3. Distribute remaining moves across classes that have more pairs
    
    Returns:
        list: [(image_path, splat_path, class_label), ...]
    """
    # Shuffle pairs within each class for random selection
    for label in pairs_by_class:
        random.shuffle(pairs_by_class[label])
    
    # Separate classes by how many pairs they have
    classes_with_multiple = []  # classes with > 1 pair
    classes_with_one = []       # classes with exactly 1 pair
    
    for label, pairs in pairs_by_class.items():
        if len(pairs) > 1:
            classes_with_multiple.append((label, pairs))
        elif len(pairs) == 1:
            classes_with_one.append((label, pairs))
    
    # Sort by number of pairs (descending) to prioritize classes with more pairs
    classes_with_multiple.sort(key=lambda x: len(x[1]), reverse=True)
    
    selected_pairs = []
    remaining_count = target_count
    
    # First pass: take 1 from each class with multiple pairs
    # This ensures we get at least 1 from as many classes as possible
    for label, pairs in classes_with_multiple:
        if remaining_count <= 0:
            break
        
        # Take 1 pair from this class
        selected_pairs.append((pairs[0][0], pairs[0][1], label))
        remaining_count -= 1
    
    # Second pass: distribute remaining moves across all classes with multiple pairs
    # We'll cycle through classes, taking 1 more from each until we reach target
    # IMPORTANT: Always leave at least 1 pair in each class
    if remaining_count > 0:
        class_idx = 0
        # Create a list of available pairs (excluding already selected ones)
        # Only include pairs that we can take while leaving at least 1 behind
        available_by_class = {}
        for label, pairs in classes_with_multiple:
            # We already took 1, so we can take up to len(pairs) - 2 more
            # (to leave at least 1 behind)
            if len(pairs) > 2:
                available_by_class[label] = pairs[1:-1]  # Exclude first (taken) and last (must keep)
            # If len(pairs) == 2, we already took 1, so we can't take more
        
        while remaining_count > 0 and len(available_by_class) > 0:
            # Get next class with available pairs
            labels = list(available_by_class.keys())
            if not labels:
                break
            
            label = labels[class_idx % len(labels)]
            available_pairs = available_by_class[label]
            
            if len(available_pairs) > 0:
                # Take 1 more pair from this class
                selected_pairs.append((available_pairs[0][0], available_pairs[0][1], label))
                available_by_class[label] = available_pairs[1:]
                remaining_count -= 1
                
                # If no more pairs available for this class, remove it
                if len(available_by_class[label]) == 0:
                    del available_by_class[label]
                    # Reset index to avoid skipping
                    class_idx = 0
                    continue
            
            class_idx += 1
    
    return selected_pairs


def move_pairs(selected_pairs, val_images_dir, val_splats_dir, dry_run=False):
    """
    Move selected pairs to validation directories.
    
    Args:
        selected_pairs: List of (image_path, splat_path, class_label) tuples
        val_images_dir: Destination directory for images
        val_splats_dir: Destination directory for splats
        dry_run: If True, only print what would be moved without actually moving
    """
    val_images_dir = Path(val_images_dir)
    val_splats_dir = Path(val_splats_dir)
    
    # Create destination directories if they don't exist
    if not dry_run:
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_splats_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    failed_count = 0
    
    for image_path, splat_path, label in tqdm(selected_pairs, desc="Moving pairs"):
        # Create class subdirectories in destination
        val_image_class_dir = val_images_dir / str(label)
        val_splat_class_dir = val_splats_dir / str(label)
        
        if not dry_run:
            val_image_class_dir.mkdir(parents=True, exist_ok=True)
            val_splat_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Destination paths
        dest_image_path = val_image_class_dir / image_path.name
        dest_splat_path = val_splat_class_dir / splat_path.name
        
        try:
            if dry_run:
                print(f"Would move: {image_path} -> {dest_image_path}")
                print(f"Would move: {splat_path} -> {dest_splat_path}")
            else:
                # Move files
                shutil.move(str(image_path), str(dest_image_path))
                shutil.move(str(splat_path), str(dest_splat_path))
            moved_count += 1
        except Exception as e:
            print(f"Error moving pair (class {label}): {e}")
            failed_count += 1
    
    return moved_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Move corresponding image-splat pairs from training to validation"
    )
    parser.add_argument(
        "--train_images_dir",
        type=str,
        default="/workspace/data/train",
        help="Source directory for training images (default: /workspace/data/train)"
    )
    parser.add_argument(
        "--train_splats_dir",
        type=str,
        default="/workspace/splats/train/nolimit",
        help="Source directory for training splats (default: /workspace/splats/train/nolimit)"
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default="/workspace/data/validation",
        help="Destination directory for validation images (default: /workspace/data/validation)"
    )
    parser.add_argument(
        "--val_splats_dir",
        type=str,
        default="/workspace/splats/validation/nolimit",
        help="Destination directory for validation splats (default: /workspace/splats/validation/nolimit)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of pairs to move (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without actually moving files"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Finding paired files...")
    print(f"  Training images: {args.train_images_dir}")
    print(f"  Training splats: {args.train_splats_dir}")
    
    pairs_by_class = find_paired_files(args.train_images_dir, args.train_splats_dir)
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_class.values())
    num_classes = len(pairs_by_class)
    
    print(f"\nFound {total_pairs} paired files across {num_classes} classes")
    
    # Count classes by size
    classes_with_one = sum(1 for pairs in pairs_by_class.values() if len(pairs) == 1)
    classes_with_multiple = num_classes - classes_with_one
    
    print(f"  Classes with 1 pair: {classes_with_one} (will be skipped)")
    print(f"  Classes with >1 pair: {classes_with_multiple}")
    
    print(f"\nSelecting {args.count} pairs to move...")
    selected_pairs = select_pairs_to_move(pairs_by_class, target_count=args.count)
    
    if len(selected_pairs) < args.count:
        print(f"Warning: Only {len(selected_pairs)} pairs available to move (requested {args.count})")
    
    # Count how many classes we're taking from
    classes_in_selection = len(set(label for _, _, label in selected_pairs))
    print(f"Selected {len(selected_pairs)} pairs from {classes_in_selection} classes")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be moved. Showing first 10 pairs that would be moved:\n")
        for i, (img_path, splat_path, label) in enumerate(selected_pairs[:10]):
            print(f"  Class {label}: {img_path.name} <-> {splat_path.name}")
        if len(selected_pairs) > 10:
            print(f"  ... and {len(selected_pairs) - 10} more pairs")
    else:
        print(f"\nMoving pairs to validation directories...")
        print(f"  Validation images: {args.val_images_dir}")
        print(f"  Validation splats: {args.val_splats_dir}")
        
        moved_count, failed_count = move_pairs(
            selected_pairs,
            args.val_images_dir,
            args.val_splats_dir,
            dry_run=False
        )
        
        print(f"\n{'='*60}")
        print(f"Move completed!")
        print(f"{'='*60}")
        print(f"Successfully moved: {moved_count} pairs")
        if failed_count > 0:
            print(f"Failed to move: {failed_count} pairs")
        print(f"Classes affected: {classes_in_selection}")


if __name__ == "__main__":
    main()

