import argparse
import os
import sys
import torch
import tqdm
from pathlib import Path
import yaml
from types import SimpleNamespace

# Add current directory to path to allow imports from Instant-GI
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add gsplat examples to path to allow importing datasets.colmap
gsplat_examples_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../gsplat/examples"))
if gsplat_examples_path not in sys.path:
    sys.path.append(gsplat_examples_path)

from generalizable_model.init_net import InitNet

try:
    from datasets.colmap import Dataset, Parser
except ImportError:
    print(f"Error: Could not import datasets.colmap from {gsplat_examples_path}")
    print("Make sure gsplat is installed and located at ../gsplat relative to Instant-GI")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D splats using Instant-GI InitNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to InitNet checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated splats")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Mip-NeRF 360 data directory")
    parser.add_argument("--factor", type=int, default=4, help="Downsample factor for Mip-NeRF 360 data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"], help="Dataset split to process")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading InitNet from {args.checkpoint}...")
    model = InitNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # Setup data loader using gsplat's COLMAP parser
    print(f"Loading data from {args.data_dir} with factor {args.factor}...")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True, # simple_trainer uses normalize=True
        test_every=8
    )
    
    if args.split == "all":
        # Combine train and val
        train_set = Dataset(parser, split="train")
        val_set = Dataset(parser, split="val")
        # We can iterate over both
        datasets = [train_set, val_set]
    else:
        datasets = [Dataset(parser, split=args.split)]
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving splats to {args.output_dir}...")
    
    count = 0
    with torch.no_grad():
        for dataset in datasets:
            # Create a loader or just iterate
            # Dataset.__getitem__ returns a dict
            
            for i in tqdm.tqdm(range(len(dataset)), desc=f"Generating Splats ({args.split})"):
                data = dataset[i]
                
                image_id = data["image_id"] # Integer ID
                image = data["image"] # [H, W, 3] tensor, 0-255
                
                # Preprocess for InitNet
                # Convert to [1, 3, H, W] float [0, 1]
                image = image.permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                
                # Run InitNet
                # get_gaussians=True returns: xy, scaling, rotation, color, triangles
                xy, _, _, _, _ = model(image, get_gaussians=True)
                
                # Save xy (2D means)
                save_path = os.path.join(args.output_dir, f"{image_id}.pt")
                torch.save(xy.cpu(), save_path)
                count += 1
            
    print(f"Finished. Generated {count} splat files.")

if __name__ == "__main__":
    main()
