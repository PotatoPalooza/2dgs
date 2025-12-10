"""
ImageNet-1k dataloader for local dataset access.
Downloads and loads ImageNet images from local storage instead of streaming from HuggingFace.
Can optionally load saved Gaussian splats for each image.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from typing import Optional

DEFAULT_FIXED_SIZE = 512


class ImageNetDataset(Dataset):
    """
    ImageNet-1k dataset loader that works with locally downloaded images.
    Can optionally load saved Gaussian splats if available.
    """
    def __init__(
        self, 
        root_dir: str, 
        split: str = "validation",
        scale: float = 1.0,
        load_gaussians: bool = False,
        gaussian_dir: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory containing ImageNet images
            split: Dataset split ("train" or "validation")
            scale: Scale factor for resizing images
            load_gaussians: Whether to load saved Gaussian splats
            gaussian_dir: Directory containing saved Gaussian splats (optional)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.scale = scale
        self.load_gaussians = load_gaussians
        self.gaussian_dir = Path(gaussian_dir) if gaussian_dir else None

        target_size = int(DEFAULT_FIXED_SIZE * self.scale)
        self.transform = transforms.Compose([
            # 1. Resize: Resizes the smaller edge of the image to 'target_size'
            # This maintains aspect ratio and prepares for cropping.
            transforms.Resize(target_size), 
            # 2. CenterCrop: Crops the image centrally to the exact fixed size,
            # ensuring all tensors in the batch are (C, H, W) where H=W=target_size.
            transforms.CenterCrop(target_size),
            # 3. ToTensor: Converts PIL image to torch.FloatTensor
            transforms.ToTensor(), 
        ])
        
        # self.transform = transforms.ToTensor()
        
        # Find all image files
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist. "
                           f"Please download ImageNet first using download_imagenet().")
        
        # Collect all image paths
        self.image_paths = []
        for ext in ['.JPEG', '.jpg', '.png']:
            self.image_paths.extend(list(split_dir.rglob(f'*{ext}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        self.image_paths = sorted(self.image_paths)
        print(f"Found {len(self.image_paths)} images in {split_dir}")
        if len(self.image_paths) < 50000 and split == "validation":
            print(f"  Note: Full ImageNet validation set has ~50k images. This appears to be a partial dataset.")
        elif len(self.image_paths) < 1280000 and split == "train":
            print(f"  Note: Full ImageNet training set has ~1.28M images. This appears to be a partial dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        # if self.scale != 1.0:
        #     image = image.resize((int(image.width * self.scale), int(image.height * self.scale)))
        
        image = self.transform(image)
        gaussians = torch.empty(0)
        
        # Optionally load Gaussian splats
        if self.load_gaussians and self.gaussian_dir:
            gaussian_path = self.gaussian_dir / f"{image_path.stem}.pth"
            if gaussian_path.exists():
                gaussians = torch.load(gaussian_path, map_location='cpu')
        
        return image, gaussians


def download_imagenet(
    save_dir: str,
    split: str = "validation",
    hf_token: Optional[str] = None,
    max_images: Optional[int] = None,
    resume: bool = True,
    image_size: Optional[int] = None
):
    """
    Download ImageNet-1k dataset from HuggingFace and save to local directory.
    Supports partial downloads and resuming interrupted downloads.
    
    Args:
        save_dir: Directory to save the downloaded images
        split: Dataset split to download ("train" or "validation")
        hf_token: HuggingFace token (if not provided, will use cached login)
        max_images: Maximum number of images to download (None for full dataset)
        resume: If True, skip images that already exist (useful for resuming)
        image_size: Optional target image size (square). If provided, images will be resized to (image_size, image_size) before saving.
                    Uses center crop to maintain aspect ratio if needed.
    
    Returns:
        Path to the downloaded dataset directory
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    split_dir = save_path / split
    split_dir.mkdir(exist_ok=True)
    
    if max_images is None:
        print(f"Downloading full ImageNet-1k {split} set to {save_path}...")
        print("This may take a while. The dataset is ~150GB for the full set.")
    else:
        print(f"Downloading {max_images} images from ImageNet-1k {split} set to {save_path}...")
    
    if image_size is not None:
        print(f"Images will be resized to {image_size}x{image_size} pixels")
    
    try:
        # Use streaming mode for partial downloads to avoid downloading everything first
        use_streaming = max_images is not None
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            streaming=use_streaming,
            token=hf_token
        )
        
        if not use_streaming:
            print(f"Downloaded dataset. Saving images to {split_dir}...")
        
        saved_count = 0
        skipped_count = 0
        
        # Save images to disk
        for i, sample in enumerate(tqdm(ds, desc=f"Saving {split} images", total=max_images)):
            # Stop if we've reached the limit
            if max_images is not None and saved_count >= max_images:
                break
            
            try:
                image = sample['image']
                label = sample['label']
                
                # Create label directory if it doesn't exist
                label_dir = split_dir / str(label)
                label_dir.mkdir(exist_ok=True)
                
                # Save image with original filename or generate one
                filename = sample.get('file_name', f"ILSVRC2012_{split}_{i:08d}.JPEG")
                # Ensure filename has proper extension
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filename = f"{filename}.JPEG"
                
                image_path = label_dir / filename
                
                # Skip if file already exists and resume is enabled
                if resume and image_path.exists():
                    skipped_count += 1
                    continue
                
                # Resize image if image_size is specified
                if image_size is not None:
                    # Resize maintaining aspect ratio, then center crop to exact size
                    # This matches the standard ImageNet preprocessing
                    width, height = image.size
                    
                    # Calculate the scaling factor to make the smaller dimension equal to image_size
                    # Then we'll center crop to get exact square
                    if width < height:
                        new_width = image_size
                        new_height = int(height * (image_size / width))
                    else:
                        new_height = image_size
                        new_width = int(width * (image_size / height))
                    
                    # Resize maintaining aspect ratio
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Center crop to exact size
                    width, height = image.size
                    left = (width - image_size) // 2
                    top = (height - image_size) // 2
                    right = left + image_size
                    bottom = top + image_size
                    image = image.crop((left, top, right, bottom))
                
                image.save(image_path)
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue
        
        print(f"Successfully saved {saved_count} images to {split_dir}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} images that already existed (resume mode)")
        return split_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you are logged in via `huggingface-cli login` and have accepted the terms.")
        raise


def get_imagenet_data_loader(
    data_path: str,
    batch_size: int,
    split: str = "validation",
    scale: float = 1.0,
    load_gaussians: bool = False,
    gaussian_dir: Optional[str] = None,
    shuffle: bool = False,
    num_workers: Optional[int] = None
):
    """
    Create a DataLoader for ImageNet dataset.
    
    Args:
        data_path: Path to ImageNet dataset root directory
        batch_size: Batch size for the DataLoader
        split: Dataset split ("train" or "validation")
        scale: Scale factor for resizing images
        load_gaussians: Whether to load saved Gaussian splats
        gaussian_dir: Directory containing saved Gaussian splats
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes (defaults to CPU count)
    
    Returns:
        DataLoader instance
    """
    dataset = ImageNetDataset(
        root_dir=data_path,
        split=split,
        scale=scale,
        load_gaussians=load_gaussians,
        gaussian_dir=gaussian_dir
    )
    
    if num_workers is None:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except:
            num_workers = os.cpu_count() or 4
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=4,
        pin_memory=True
    )
