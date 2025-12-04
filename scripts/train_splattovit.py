#!/usr/bin/env python3
"""
Train SplatToViT model on ImageNet splat data.

This script loads Gaussian splat .pt files from an ImageNet directory structure,
handles variable numbers of splats per image through padding and attention masks,
and trains the SplatToViT model.

Usage:
    python scripts/train_splattovit.py \
        --train_dir ./data/imagenet_splats/train \
        --val_dir ./data/imagenet_splats/validation \
        --output_dir ./output/splattovit \
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.splattovit import SplatToViT


class ImageNetSplatDataset(Dataset):
    """
    Dataset for loading Gaussian splat .pt files from ImageNet directory structure.
    
    Expects directory structure:
        root_dir/
            0/  (class label directories)
                img1.pt
                img2.pt
                ...
            1/
                ...
    """
    
    def __init__(
        self, 
        root_dir: Path,
        max_splats: Optional[int] = None,
        splat_limit_method: str = "random"
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing class label subdirectories with .pt files
            max_splats: Optional maximum number of splats per image. If None, use all splats.
            splat_limit_method: Method to use when limiting splats. Options: "random", "opacity", "importance"
        """
        self.root_dir = Path(root_dir)
        self.max_splats = max_splats
        self.splat_limit_method = splat_limit_method
        
        # Validate limit method
        if self.max_splats is not None and self.max_splats <= 0:
            raise ValueError(f"max_splats must be positive, got {self.max_splats}")
        if self.splat_limit_method not in ["random", "opacity", "importance"]:
            raise ValueError(
                f"splat_limit_method must be one of ['random', 'opacity', 'importance'], "
                f"got {self.splat_limit_method}"
            )
        
        if not self.root_dir.exists():
            raise ValueError(f"Data directory {self.root_dir} does not exist")
        
        # Collect all .pt files with their labels
        self.splat_paths = []
        self.labels = []
        
        # Find all class directories (should be numeric)
        for label_dir in sorted(self.root_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            try:
                label = int(label_dir.name)
            except ValueError:
                # Skip non-numeric directories
                continue
            
            # Find all .pt files in this label directory
            for splat_path in sorted(label_dir.glob("*.pt")):
                self.splat_paths.append(splat_path)
                self.labels.append(label)
        
        if len(self.splat_paths) == 0:
            raise ValueError(f"No .pt files found in {self.root_dir}")
        
        # Get unique labels and create label mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Found {len(self.splat_paths)} splat files")
        print(f"Number of classes: {self.num_classes}")
        print(f"Label range: {min(self.labels)} to {max(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.splat_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Load a splat file and return splat data with label.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (splat_data dict, label_index)
        """
        splat_path = self.splat_paths[idx]
        label_idx = self.label_indices[idx]
        
        # Load splat data from .pt file
        splat_data = torch.load(splat_path, map_location='cpu')
        
        # Extract required fields: xy, scaling, rotation, color
        # Note: triangles are optional and not needed for training
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
        
        return splat_dict, label_idx


def collate_splats(
    batch: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Collate function that pads splats to the maximum number in the batch.
    
    Args:
        batch: List of (splat_data, label) tuples
    
    Returns:
        Tuple of:
            - batched_splat_data: Dictionary with padded tensors of shape (B, M_max, ...)
            - attention_mask: Boolean tensor of shape (B, M_max) where 1 = real splat, 0 = padding
            - labels: Tensor of shape (B,) with label indices
    """
    splat_dicts, labels = zip(*batch)
    
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
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return batched_splat_data, attention_mask, labels_tensor


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
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (splat_data, attention_mask, labels) in enumerate(pbar):
        # Move to device
        splat_data = {k: v.to(device) for k, v in splat_data.items()}
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(splat_data, attention_mask=attention_mask)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
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
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        for batch_idx, (splat_data, attention_mask, labels) in enumerate(pbar):
            # Move to device
            splat_data = {k: v.to(device) for k, v in splat_data.items()}
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(splat_data, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SplatToViT model on ImageNet splat data"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing training split ImageNet splat .pt files with class label subdirectories"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Directory containing validation split ImageNet splat .pt files with class label subdirectories"
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
        default=2e-4,
        help="Learning rate (default: 2e-4)"
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
        "--vit_model",
        type=str,
        default="vit_base_patch16_224",
        help="ViT model name from timm (default: vit_base_patch16_224)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained ViT weights (default: True)"
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Don't use pretrained ViT weights"
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
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file containing adapter state (latent_vectors and adapter.* keys). The patch_projection.* keys will be ignored."
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
    print(f"\nLoading training dataset from {args.train_dir}...")
    train_dataset = ImageNetSplatDataset(
        Path(args.train_dir),
        max_splats=args.max_splats,
        splat_limit_method=args.limit_method
    )
    num_classes = train_dataset.num_classes
    print(f"Training dataset loaded: {len(train_dataset)} samples, {num_classes} classes")
    if args.max_splats is not None:
        print(f"  Max splats per image: {args.max_splats} (method: {args.limit_method})")
    
    # Load validation dataset
    print(f"\nLoading validation dataset from {args.val_dir}...")
    val_dataset = ImageNetSplatDataset(
        Path(args.val_dir),
        max_splats=args.max_splats,
        splat_limit_method=args.limit_method
    )
    # Ensure validation dataset uses same label mapping as training
    val_dataset.label_to_idx = train_dataset.label_to_idx
    val_dataset.idx_to_label = train_dataset.idx_to_label
    val_dataset.num_classes = train_dataset.num_classes
    # Convert validation labels to match training label indices
    val_dataset.label_indices = [
        train_dataset.label_to_idx.get(label, 0) 
        for label in val_dataset.labels
    ]
    print(f"Validation dataset loaded: {len(val_dataset)} samples, {num_classes} classes")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_splats,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_splats,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    print(f"\nCreating SplatToViT model...")
    model = SplatToViT(
        latent_seq_len=args.latent_seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        vit_model_name=args.vit_model,
        pretrained=args.pretrained,
        num_classes=num_classes,
    ).to(device)
    
    print(f"Model created:")
    print(f"  Latent seq len: {args.latent_seq_len}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  ViT model: {args.vit_model}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Num classes: {num_classes}")
    
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

    criterion = nn.CrossEntropyLoss()
    
    # Load adapter checkpoint if provided
    if args.adapter_checkpoint is not None:
        checkpoint_path = Path(args.adapter_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Adapter checkpoint file not found: {checkpoint_path}")
        
        print(f"\nLoading adapter checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            checkpoint_state_dict = checkpoint['model_state_dict']
        else:
            # If checkpoint is just the state dict itself
            checkpoint_state_dict = checkpoint
        
        # Filter to only adapter-related keys (latent_vectors and adapter.*)
        # and exclude patch_projection.* keys
        model_state_dict = model.state_dict()
        adapter_state_dict = {}
        
        for key in checkpoint_state_dict.keys():
            # Include latent_vectors and adapter.* keys
            # Exclude patch_projection.* keys
            if key == 'latent_vectors' or key.startswith('adapter.'):
                if key in model_state_dict:
                    # Verify shapes match
                    if checkpoint_state_dict[key].shape == model_state_dict[key].shape:
                        adapter_state_dict[key] = checkpoint_state_dict[key]
                        print(f"  Loaded: {key} {checkpoint_state_dict[key].shape}")
                    else:
                        print(f"  Warning: Shape mismatch for {key}. Checkpoint: {checkpoint_state_dict[key].shape}, Model: {model_state_dict[key].shape}. Skipping.")
                else:
                    print(f"  Warning: Key {key} not found in model. Skipping.")
            elif key.startswith('patch_projection.'):
                print(f"  Ignoring: {key}")
            # Silently ignore other keys
        
        # Load the filtered adapter state dict
        # missing_keys, unexpected_keys = model.load_state_dict(adapter_state_dict, strict=False)
        
        # if missing_keys:
        #     print(f"  Warning: Some adapter keys were not loaded: {missing_keys}")
        # if unexpected_keys:
        #     print(f"  Warning: Some unexpected keys in checkpoint: {unexpected_keys}")
        
        print(f"  Loaded {len(adapter_state_dict)} adapter parameters")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Epochs: {args.epochs}")
    if args.max_splats is not None:
        print(f"  Max splats per image: {args.max_splats} (method: {args.limit_method})")
    if args.adapter_checkpoint is not None:
        print(f"  Loading adapter from checkpoint: {args.adapter_checkpoint}")
    
    # Save initial model checkpoint (epoch 0) before training
    print(f"\nSaving initial model checkpoint (epoch 0)...")
    checkpoint_path_epoch0 = output_dir / "checkpoint_epoch0.pt"
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': {},
        'val_metrics': {},
        'args': vars(args),
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
    }, checkpoint_path_epoch0)
    print(f"✓ Saved initial model to {checkpoint_path_epoch0}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
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
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint if best validation accuracy
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': vars(args),
            'num_classes': num_classes,
            'best_val_acc': best_val_acc,
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
                'num_classes': num_classes,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
            }, best_checkpoint_path)
            print(f"✓ Saved best model to {best_checkpoint_path}")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Best model saved to: {output_dir / 'checkpoint_best.pt'}")
    print(f"\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()

