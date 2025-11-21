import argparse
import os
import sys
import torch
import tqdm
import numpy as np
import imageio.v2 as imageio
import cv2
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add current directory to path to allow imports from Instant-GI
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generalizable_model.init_net import InitNet

try:
    from pycolmap import SceneManager
except ImportError:
    print("Error: pycolmap not found. Please install it via 'pip install pycolmap'.")
    sys.exit(1)

# --- Minimal COLMAP Loader Classes ---

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm.tqdm(image_files, desc="Resizing images"):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir

class MinimalParser:
    """Minimal COLMAP parser for Instant-GI generation."""

    def __init__(self, data_dir: str, factor: int = 1):
        self.data_dir = data_dir
        self.factor = factor

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()

        imdata = manager.images
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        
        # Sort images by name to match gsplat's ordering
        image_names = [imdata[k].name for k in imdata]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        
        # Re-order keys based on sorted names
        sorted_keys = [k for k in imdata]
        sorted_keys = [sorted_keys[i] for i in inds]

        for k in sorted_keys:
            im = imdata[k]
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            # Mapping from pycolmap camera types to params
            # 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL, 4: OPENCV, 5: OPENCV_FISHEYE
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            else:
                # Fallback or error
                print(f"Warning: Unknown camera type {type_}, assuming perspective with no distortion.")
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        # Handle images
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        
        if not os.path.exists(colmap_image_dir):
             raise ValueError(f"Image folder {colmap_image_dir} does not exist.")

        # Resize if needed
        if factor > 1 and not os.path.exists(image_dir):
             _resize_image_folder(colmap_image_dir, image_dir, factor=factor)
        elif factor > 1 and os.path.exists(image_dir):
             # Check if we need to resize (simple check: if empty or mismatch count)
             pass 

        # Map colmap images to actual images
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        
        # If resizing happened, we might have pngs instead of jpgs
        # Simple mapping: assume sorted order matches
        if len(colmap_files) != len(image_files):
             print("Warning: Number of files in images/ and resized images/ mismatch.")
             
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        self.image_names = image_names
        self.image_paths = image_paths
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        
        # Pre-compute undistortion maps
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            
            # Assuming perspective for simplicity as per gsplat code mostly handling this
            # If fisheye, we need that logic too.
            # Copying logic from gsplat/examples/datasets/colmap.py
            
            # Determine camtype again or store it
            # For brevity, let's infer or assume perspective if not fisheye params
            # But we need to be robust.
            
            # Re-checking type logic is a bit redundant, let's just assume standard opencv undistort works for most
            # unless it's fisheye.
            # To be safe, let's implement the standard perspective undistortion which covers most cases.
            
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist


class MinimalDataset:
    def __init__(self, parser: MinimalParser, split: str = "train", test_every: int = 8):
        self.parser = parser
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % test_every != 0]
        elif split == "val":
            self.indices = indices[indices % test_every == 0]
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        image_path = self.parser.image_paths[index]
        image = imageio.imread(image_path)[..., :3]
        camera_id = self.parser.camera_ids[index]
        params = self.parser.params_dict[camera_id]

        if len(params) > 0 and camera_id in self.parser.mapx_dict:
            # Undistort
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            if camera_id in self.parser.roi_undist_dict:
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]

        return {
            "image": torch.from_numpy(image).float(), # [H, W, 3]
            "image_id": index
        }

# --- Main Script ---

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
    
    # Setup data loader using MinimalParser
    print(f"Loading data from {args.data_dir} with factor {args.factor}...")
    parser = MinimalParser(
        data_dir=args.data_dir,
        factor=args.factor
    )
    
    if args.split == "all":
        # Combine train and val
        train_set = MinimalDataset(parser, split="train")
        val_set = MinimalDataset(parser, split="val")
        datasets = [train_set, val_set]
    else:
        datasets = [MinimalDataset(parser, split=args.split)]
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving splats to {args.output_dir}...")
    
    count = 0
    with torch.no_grad():
        for dataset in datasets:
            for i in tqdm.tqdm(range(len(dataset)), desc=f"Generating Splats"):
                data = dataset[i]
                
                image_id = data["image_id"] # Integer ID
                image = data["image"] # [H, W, 3] tensor, 0-255
                
                # Preprocess for InitNet
                # Convert to [1, 3, H, W] float [0, 1]
                image = image.permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                
                # Run InitNet
                xy, _, _, _, _ = model(image, get_gaussians=True)
                
                # Save xy (2D means)
                save_path = os.path.join(args.output_dir, f"{image_id}.pt")
                torch.save(xy.cpu(), save_path)
                count += 1
            
    print(f"Finished. Generated {count} splat files.")

if __name__ == "__main__":
    main()
