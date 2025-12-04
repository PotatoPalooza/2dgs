#!/usr/bin/env python3
"""
Train GaussianRenderer model to reconstruct images from splats.

This script loads Gaussian splat .pt files and corresponding original images from ImageNet
directory structure, and trains the renderer to reconstruct images from splats using 
combined L1 and perceptual loss.

Usage:
    python scripts/train_renderer.py \
        --train_splats_dir ./data/imagenet_splats/train \
        --train_images_dir ./data/imagenet/train \
        --val_splats_dir ./data/imagenet_splats/validation \
        --val_images_dir ./data/imagenet/validation \
        --output_dir ./output/renderer \
        --batch_size 32 \
        --lr 1e-4 \
        --epochs 10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.renderer import GaussianRenderer


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Computes L1 loss between feature representations from a pre-trained VGG network.
    """
    
    def __init__(self, feature_layers=None, weights=None):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: List of layer indices to extract features from.
                          If None, uses default layers [3, 8, 15, 22] (relu1_2, relu2_2, relu3_3, relu4_3).
            weights: Optional weights for each feature layer. If None, uses equal weights.
        """
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        features.eval()  # Set to eval mode to disable dropout and use eval batch norm
        
        # Default feature layers: relu1_2, relu2_2, relu3_3, relu4_3
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]
        
        self.feature_layers = feature_layers
        self.feature_extractor = nn.ModuleList([features[i] for i in range(max(feature_layers) + 1)])
        
        # Freeze VGG parameters and ensure eval mode
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        # Set weights for each layer (default: equal weights)
        if weights is None:
            weights = [1.0 / len(feature_layers)] * len(feature_layers)
        self.weights = weights
        
        # Normalize input to match ImageNet preprocessing (mean and std)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input to match ImageNet preprocessing."""
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        """Extract features from multiple layers of VGG."""
        # Ensure feature extractor is in eval mode (important when model is in training mode)
        # This ensures batch norm uses running stats and dropout is disabled
        was_training = self.feature_extractor.training
        if was_training:
            self.feature_extractor.eval()
        
        x = self.normalize(x)
        features = []
        # Allow gradients to flow through (but VGG params have requires_grad=False)
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        # Restore original training state
        if was_training:
            self.feature_extractor.train(was_training)
        
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image tensor of shape (B, 3, H, W) in range [0, 1]
            target: Target image tensor of shape (B, 3, H, W) in range [0, 1]
        
        Returns:
            Perceptual loss value
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function: L1 loss + perceptual loss.
    """
    
    def __init__(self, l1_weight=1.0, perceptual_weight=1.0, feature_layers=None, perceptual_weights=None):
        """
        Initialize combined loss.
        
        Args:
            l1_weight: Weight for L1 loss (default: 1.0)
            perceptual_weight: Weight for perceptual loss (default: 1.0)
            feature_layers: Feature layers for perceptual loss (default: None, uses default)
            perceptual_weights: Weights for each perceptual feature layer (default: None, uses equal weights)
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(feature_layers=feature_layers, weights=perceptual_weights)
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(self, pred, target):
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image tensor of shape (B, 3, H, W)
            target: Target image tensor of shape (B, 3, H, W)
        
        Returns:
            Combined loss value
        """
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        return self.l1_weight * l1 + self.perceptual_weight * perceptual


class ImageNetSplatImageDataset(Dataset):
    """
    Dataset for loading paired Gaussian splat .pt files and corresponding images.
    
    Expects directory structure:
        splats_dir/
            0/  (class label directories)
                img1.pt
                img2.pt
                ...
        images_dir/
            0/
                img1.jpeg
                img2.jpeg
                ...
    """
    
    def __init__(
        self, 
        splats_dir: Path,
        images_dir: Path,
        max_splats: Optional[int] = None,
        splat_limit_method: str = "random",
        image_size: int = 224,
    ):
        """
        Initialize dataset.
        
        Args:
            splats_dir: Root directory containing class label subdirectories with .pt files
            images_dir: Root directory containing class label subdirectories with image files
            max_splats: Optional maximum number of splats per image. If None, use all splats.
            splat_limit_method: Method to use when limiting splats. Options: "random", "opacity", "importance"
            image_size: Size to resize images to (default: 224)
        """
        self.splats_dir = Path(splats_dir)
        self.images_dir = Path(images_dir)
        self.max_splats = max_splats
        self.splat_limit_method = splat_limit_method
        self.image_size = image_size
        
        # Validate limit method
        if self.max_splats is not None and self.max_splats <= 0:
            raise ValueError(f"max_splats must be positive, got {self.max_splats}")
        if self.splat_limit_method not in ["random", "opacity", "importance"]:
            raise ValueError(
                f"splat_limit_method must be one of ['random', 'opacity', 'importance'], "
                f"got {self.splat_limit_method}"
            )
        
        if not self.splats_dir.exists():
            raise ValueError(f"Splats directory {self.splats_dir} does not exist")
        if not self.images_dir.exists():
            raise ValueError(f"Images directory {self.images_dir} does not exist")
        
        # Image transform: resize and convert to tensor [0, 1]
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Collect all .pt files with their corresponding image paths
        self.splat_paths = []
        self.image_paths = []
        
        # Find all class directories (should be numeric)
        for label_dir in sorted(self.splats_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            try:
                label = int(label_dir.name)
            except ValueError:
                # Skip non-numeric directories
                continue
            
            # Find corresponding image directory
            image_label_dir = self.images_dir / label_dir.name
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
                
                self.splat_paths.append(splat_path)
                self.image_paths.append(image_path)
        
        if len(self.splat_paths) == 0:
            raise ValueError(f"No paired splat/image files found in {self.splats_dir} and {self.images_dir}")
        
        print(f"Found {len(self.splat_paths)} paired splat/image files")
    
    def __len__(self) -> int:
        return len(self.splat_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load a splat file and corresponding image.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (splat_data dict, image_tensor)
        """
        splat_path = self.splat_paths[idx]
        image_path = self.image_paths[idx]
        
        # Load splat data from .pt file
        splat_data = torch.load(splat_path, map_location='cpu')
        
        # Extract required fields: xy, scaling, rotation, color
        splat_dict = {
            'xy': splat_data['xy'],  # (M, 2)
            'scaling': splat_data['scaling'],  # (M, 2)
            'rotation': splat_data['rotation'],  # (M, 1)
            'color': splat_data['color'],  # (M, 3)
        }
        
        # Apply splat limit if specified
        if self.max_splats is not None:
            num_splats = splat_dict['xy'].shape[0]
            if num_splats > self.max_splats:
                if self.splat_limit_method == "random":
                    # Randomly sample indices
                    indices = torch.randperm(num_splats)[:self.max_splats]
                elif self.splat_limit_method == "opacity":
                    # Sort by average scaling (larger = more important)
                    avg_scaling = splat_dict['scaling'].mean(dim=1)
                    _, indices = torch.topk(avg_scaling, self.max_splats)
                elif self.splat_limit_method == "importance":
                    # Combine scaling and color intensity as importance metric
                    avg_scaling = splat_dict['scaling'].mean(dim=1)
                    color_intensity = splat_dict['color'].mean(dim=1)
                    importance = avg_scaling * color_intensity
                    _, indices = torch.topk(importance, self.max_splats)
                else:
                    raise ValueError(f"Unknown splat_limit_method: {self.splat_limit_method}")
                
                # Apply indices to all splat fields
                for key in splat_dict:
                    splat_dict[key] = splat_dict[key][indices]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)  # (3, H, W) in range [0, 1]
        
        return splat_dict, image_tensor


def collate_splats_and_images(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Collate function that pads splats to the maximum number in the batch.
    
    Args:
        batch: List of (splat_data, image) tuples
    
    Returns:
        Tuple of:
            - batched_splat_data: Dictionary with padded tensors of shape (B, M_max, ...)
            - attention_mask: Boolean tensor of shape (B, M_max) where 1 = real splat, 0 = padding
            - images: Tensor of shape (B, 3, H, W) with ground truth images
    """
    splat_dicts, images = zip(*batch)
    
    # Stack images: (B, 3, H, W)
    images_tensor = torch.stack(images, dim=0)
    
    # Find maximum number of splats in this batch
    num_splats = [splat_dict['xy'].shape[0] for splat_dict in splat_dicts]
    max_splats = max(num_splats)
    batch_size = len(batch)
    
    # Create attention mask: 1 for real splats, 0 for padding
    attention_mask = torch.zeros(batch_size, max_splats, dtype=torch.bool)
    for i, num in enumerate(num_splats):
        attention_mask[i, :num] = True
    
    # Pad all splat tensors to max_splats
    batched_splat_data = {}
    for key in ['xy', 'scaling', 'rotation', 'color']:
        tensors = []
        for i, splat_dict in enumerate(splat_dicts):
            tensor = splat_dict[key]  # (M_i, ...)
            num = num_splats[i]
            
            # Get feature dimensions
            if key == 'xy':
                feature_dim = 2
            elif key == 'scaling':
                feature_dim = 2
            elif key == 'rotation':
                feature_dim = 1
            elif key == 'color':
                feature_dim = 3
            else:
                raise ValueError(f"Unknown key: {key}")
            
            # Pad with zeros to max_splats
            if num < max_splats:
                padding = torch.zeros(max_splats - num, feature_dim, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=0)
            
            tensors.append(tensor)
        
        # Stack into batch: (B, M_max, feature_dim)
        batched_splat_data[key] = torch.stack(tensors, dim=0)
    
    return batched_splat_data, attention_mask, images_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (splat_data, attention_mask, images) in enumerate(pbar):
        # Move to device
        splat_data = {k: v.to(device) for k, v in splat_data.items()}
        attention_mask = attention_mask.to(device)
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        rendered_images = model(splat_data, attention_mask=attention_mask)
        
        # Compute loss (MSE between rendered and ground truth)
        loss = criterion(rendered_images, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
        })
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        for batch_idx, (splat_data, attention_mask, images) in enumerate(pbar):
            # Move to device
            splat_data = {k: v.to(device) for k, v in splat_data.items()}
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            
            # Forward pass
            rendered_images = model(splat_data, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(rendered_images, images)
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
            })
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train GaussianRenderer model to reconstruct images from splats"
    )
    parser.add_argument(
        "--train_splats_dir",
        type=str,
        required=True,
        help="Directory containing training split ImageNet splat .pt files with class label subdirectories"
    )
    parser.add_argument(
        "--train_images_dir",
        type=str,
        required=True,
        help="Directory containing training split ImageNet images with class label subdirectories"
    )
    parser.add_argument(
        "--val_splats_dir",
        type=str,
        required=True,
        help="Directory containing validation split ImageNet splat .pt files with class label subdirectories"
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        required=True,
        help="Directory containing validation split ImageNet images with class label subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for saving checkpoints and logs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--latent_seq_len",
        type=int,
        default=196,
        help="Number of latent tokens (default: 196)"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size in pixels (default: 16)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size in pixels (default: 224)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--max_splats",
        type=int,
        default=None,
        help="Optional maximum number of splats per image. If not specified, use all splats."
    )
    parser.add_argument(
        "--limit_method",
        type=str,
        default="random",
        choices=["random", "opacity", "importance"],
        help="Method to use when limiting splats: 'random' (random selection), 'opacity' (by scaling/opacity), or 'importance' (by scaling * color intensity). Default: 'random'"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from. If provided, training will start from the saved epoch."
    )
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=1.0,
        help="Weight for L1 loss component (default: 1.0)"
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=1.0,
        help="Weight for perceptual loss component (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training dataset
    print(f"\nLoading training dataset...")
    print(f"  Splats: {args.train_splats_dir}")
    print(f"  Images: {args.train_images_dir}")
    train_dataset = ImageNetSplatImageDataset(
        splats_dir=Path(args.train_splats_dir),
        images_dir=Path(args.train_images_dir),
        max_splats=args.max_splats,
        splat_limit_method=args.limit_method,
        image_size=args.image_size,
    )
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    if args.max_splats is not None:
        print(f"  Max splats per image: {args.max_splats} (method: {args.limit_method})")
    
    # Load validation dataset
    print(f"\nLoading validation dataset...")
    print(f"  Splats: {args.val_splats_dir}")
    print(f"  Images: {args.val_images_dir}")
    val_dataset = ImageNetSplatImageDataset(
        splats_dir=Path(args.val_splats_dir),
        images_dir=Path(args.val_images_dir),
        max_splats=args.max_splats,
        splat_limit_method=args.limit_method,
        image_size=args.image_size,
    )
    print(f"Validation dataset loaded: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_splats_and_images,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_splats_and_images,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    print(f"\nCreating GaussianRenderer model...")
    model = GaussianRenderer(
        latent_seq_len=args.latent_seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        image_size=args.image_size,
    ).to(device)
    
    print(f"Model created:")
    print(f"  Latent seq len: {args.latent_seq_len}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Image size: {args.image_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)
    
    # Load checkpoint if provided
    start_epoch = 1
    if args.resume is not None:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        # Ensure model is on the correct device
        model = model.to(device)
        print(f"  Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"  Optimizer state loaded")
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Previous best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    else:
        best_val_loss = float('inf')
        best_epoch = 0
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Loss: Combined (L1 weight: {args.l1_weight}, Perceptual weight: {args.perceptual_weight})")
    if args.max_splats is not None:
        print(f"  Max splats per image: {args.max_splats} (method: {args.limit_method})")
    if args.resume is not None:
        print(f"  Resuming from checkpoint: {args.resume}")
        print(f"  Starting from epoch: {start_epoch}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_metrics['loss']:.6f}")
        print(f"Val   - Loss: {val_metrics['loss']:.6f}")
        
        # Save checkpoint if best validation loss
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            print(f"✓ New best validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': vars(args),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        }, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_checkpoint_path = output_dir / "checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, best_checkpoint_path)
            print(f"✓ Saved best model to {best_checkpoint_path}")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Best model saved to: {output_dir / 'checkpoint_best.pt'}")
    print(f"\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()

