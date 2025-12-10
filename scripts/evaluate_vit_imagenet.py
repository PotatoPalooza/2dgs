#!/usr/bin/env python3
"""
Evaluate a pretrained Vision Transformer (ViT) on ImageNet-1k validation set.

Usage:
    python scripts/evaluate_vit_imagenet.py --data_dir ./data/imagenet --model_name vit_base_patch16_224
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class ImageNetDatasetWithLabels(Dataset):
    """
    ImageNet-1k dataset loader that returns images and labels.
    Labels are extracted from the directory structure (validation/0/, validation/1/, etc.)
    """
    def __init__(
        self, 
        root_dir: str, 
        split: str = "validation",
        image_size: int = 224
    ):
        """
        Args:
            root_dir: Root directory containing ImageNet images
            split: Dataset split ("train" or "validation")
            image_size: Target image size for the model (default 224 for ViT)
        """
        self.root_dir = Path(root_dir)
        # self.split = split
        self.image_size = image_size

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Find all image files and extract labels from directory structure
        split_dir = self.root_dir
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist. "
                           f"Please download ImageNet first using download_imagenet().")
        
        # Collect all image paths with their labels
        self.image_paths = []
        self.labels = []
        
        # Images are stored in label directories: validation/0/, validation/1/, etc.
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            try:
                label = int(label_dir.name)
            except ValueError:
                # Skip non-numeric directories
                continue
            
            # Find all images in this label directory
            for ext in ['.JPEG', '.jpg', '.png']:
                for image_path in label_dir.glob(f'*{ext}'):
                    self.image_paths.append(image_path)
                    self.labels.append(label)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        # Sort by path for consistency
        sorted_pairs = sorted(zip(self.image_paths, self.labels))
        self.image_paths, self.labels = zip(*sorted_pairs)
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)
        
        print(f"Found {len(self.image_paths)} images in {split_dir}")
        print(f"Number of classes: {len(set(self.labels))}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return image, label


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate the model on the dataset.
    
    Returns:
        top1_acc: Top-1 accuracy
        top5_acc: Top-5 accuracy
    """
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate top-1 and top-5 accuracy
        _, predicted_top1 = outputs.topk(1, dim=1)
        _, predicted_top5 = outputs.topk(5, dim=1)
        
        correct_top1 += (predicted_top1.squeeze() == labels).sum().item()
        correct_top5 += (predicted_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    
    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained ViT on ImageNet-1k validation set"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to ImageNet dataset root directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_base_patch16_224",
        help="ViT model name from timm (e.g., 'vit_base_patch16_224', 'vit_large_patch16_224')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU count)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (default: True)"
    )
    parser.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="Don't use pretrained weights"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {args.pretrained}")
    print()
    
    # Load model
    print(f"Loading model '{args.model_name}'...")
    try:
        model = timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=1000  # ImageNet-1k has 1000 classes
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Available ViT models in timm:")
        vit_models = [m for m in timm.list_models() if 'vit' in m.lower()]
        for m in sorted(vit_models)[:20]:  # Show first 20
            print(f"  - {m}")
        if len(vit_models) > 20:
            print(f"  ... and {len(vit_models) - 20} more")
        sys.exit(1)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully!")
    print()
    
    # Create dataset and dataloader
    print("Loading ImageNet validation dataset...")
    dataset = ImageNetDatasetWithLabels(
        root_dir=args.data_dir,
        split="validation",
        image_size=224  # Standard ViT input size
    )
    
    if args.num_workers is None:
        import os
        try:
            num_workers = len(os.sched_getaffinity(0))
        except:
            num_workers = os.cpu_count() or 4
    else:
        num_workers = args.num_workers
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print()
    
    # Evaluate
    print("Starting evaluation...")
    top1_acc, top5_acc = evaluate(model, dataloader, device)
    
    # Print results
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("=" * 60)
    print()
    print("✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

