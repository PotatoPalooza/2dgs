import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np


def _cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ImageAndPF(Dataset):
    def __init__(self, root_dir, scale=1.0):
        self.root_dir = root_dir
        self.transform = transforms.ToTensor()
        self.imgs = os.listdir(root_dir)
        self.scale = scale

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        img_dir = os.path.join(self.root_dir, image_name)
        image = Image.open(os.path.join(img_dir, "gt.png"))
        if self.scale != 1.0:
            image = image.resize((int(image.width * self.scale), int(image.height * self.scale)))

        image = self.transform(image)
        gt_pf_path = os.path.join(img_dir, "gt_pf.npy")
        gt_pf = torch.tensor(np.load(gt_pf_path)).float()
        if self.scale != 1.0:
            gt_pf = F.interpolate(
                gt_pf.unsqueeze(0).unsqueeze(0),
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
        return image, gt_pf


class Div2KPatchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        scale=1.0,
        patch_size=None,
        scales=None,
        patches_per_image=1,
        samples_per_epoch=None,
    ):
        if patch_size is None:
            raise ValueError("patch_size must be specified for Div2KPatchDataset")
        self.root_dir = root_dir
        self.base_scale = scale
        self.scale_choices = self._normalize_scales(scales)
        if not self.scale_choices:
            self.scale_choices = [float(self.base_scale)]
        self.patch_h, self.patch_w = self._normalize_patch_size(patch_size)
        self.transform = transforms.ToTensor()
        self.sample_dirs = sorted(
            [
                os.path.join(root_dir, entry)
                for entry in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, entry))
            ]
        )
        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No samples found under {root_dir}")
        self.patches_per_image = max(1, int(patches_per_image))
        self.samples_per_epoch = (
            max(1, int(samples_per_epoch)) if samples_per_epoch is not None else None
        )

    @staticmethod
    def _normalize_patch_size(patch_size):
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2:
                raise ValueError("patch_size sequence must be (height, width)")
            h, w = int(patch_size[0]), int(patch_size[1])
        else:
            h = w = int(patch_size)
        if h <= 0 or w <= 0:
            raise ValueError("patch_size must be positive")
        return h, w

    @staticmethod
    def _normalize_scales(scales):
        if not scales:
            return None
        normalized = []
        for scale in scales:
            scale_val = float(scale)
            if scale_val <= 0:
                raise ValueError("scale factors must be positive")
            normalized.append(scale_val)
        return normalized

    @staticmethod
    def _pad_to_size(img_tensor, pf_tensor, target_h, target_w):
        _, cur_h, cur_w = img_tensor.shape
        if cur_h == target_h and cur_w == target_w:
            return img_tensor, pf_tensor
        pad_h = max(0, target_h - cur_h)
        pad_w = max(0, target_w - cur_w)
        if pad_h == 0 and pad_w == 0:
            return img_tensor, pf_tensor
        img_tensor = F.pad(
            img_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
        ).squeeze(0)
        pf_tensor = F.pad(
            pf_tensor.unsqueeze(0).unsqueeze(0),
            (0, pad_w, 0, pad_h),
            mode="replicate"
        ).squeeze(0).squeeze(0)
        return img_tensor, pf_tensor

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.sample_dirs) * self.patches_per_image * len(self.scale_choices)

    def _load_image(self, sample_dir):
        image_path = os.path.join(sample_dir, "gt.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing gt.png under {sample_dir}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return image

    def _load_pf(self, sample_dir):
        pf_path = os.path.join(sample_dir, "gt_pf.npy")
        if not os.path.exists(pf_path):
            raise FileNotFoundError(f"Missing gt_pf.npy under {sample_dir}")
        gt_pf = torch.tensor(np.load(pf_path)).float()
        return gt_pf

    def _resize_pf(self, gt_pf, height, width):
        if gt_pf.shape[-2] == height and gt_pf.shape[-1] == width:
            return gt_pf
        gt_pf = gt_pf.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            gt_pf,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )
        return resized.squeeze(0).squeeze(0)

    def _resolve_indices(self, idx):
        if self.samples_per_epoch is not None:
            img_idx = torch.randint(0, len(self.sample_dirs), (1,)).item()
            scale_idx = torch.randint(0, len(self.scale_choices), (1,)).item()
            return img_idx, scale_idx
        total_scales = len(self.scale_choices)
        scale_idx = idx % total_scales
        patch_slot = idx // total_scales
        img_idx = patch_slot % len(self.sample_dirs)
        return img_idx, scale_idx

    def __getitem__(self, idx):
        img_idx, scale_idx = self._resolve_indices(idx)
        sample_dir = self.sample_dirs[img_idx]
        image = self._load_image(sample_dir)
        gt_pf = self._load_pf(sample_dir)
        scale = self.scale_choices[scale_idx]
        target_w = max(1, int(round(image.width * scale)))
        target_h = max(1, int(round(image.height * scale)))
        if target_w != image.width or target_h != image.height:
            image = image.resize((target_w, target_h), Image.BICUBIC)
            gt_pf = self._resize_pf(gt_pf, target_h, target_w)
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size ({width}x{height}) for {sample_dir}")
        crop_w = min(self.patch_w, width)
        crop_h = min(self.patch_h, height)
        max_left = max(0, width - crop_w)
        max_top = max(0, height - crop_h)
        left = 0 if max_left == 0 else torch.randint(0, max_left + 1, (1,)).item()
        top = 0 if max_top == 0 else torch.randint(0, max_top + 1, (1,)).item()
        right = min(left + crop_w, width)
        bottom = min(top + crop_h, height)
        cropped_image = image.crop((left, top, right, bottom))
        image_tensor = self.transform(cropped_image)
        pf_patch = gt_pf[top:bottom, left:right]
        image_tensor, pf_patch = self._pad_to_size(
            image_tensor, pf_patch, self.patch_h, self.patch_w
        )
        return image_tensor, pf_patch


def pad_collate(batch):
    images, pfs = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    padded_pfs = []
    for img, pf in zip(images, pfs):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Negative padding encountered.")
        padded_images.append(
            F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        )
        padded_pfs.append(
            F.pad(pf.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
        )

    stacked_images = torch.stack(padded_images, dim=0)
    stacked_pfs = torch.stack(padded_pfs, dim=0)
    return stacked_images, stacked_pfs


def get_data_loader(data_path, batch_size, scale=1.0, patch_config=None):
    patch_enabled = bool(_cfg_value(patch_config, "enabled", False))
    if patch_enabled:
        patch_size = _cfg_value(patch_config, "patch_size", None)
        if patch_size is None:
            raise ValueError("patch_size must be specified when patch sampling is enabled")
        dataset = Div2KPatchDataset(
            data_path,
            scale=scale,
            patch_size=patch_size,
            scales=_cfg_value(patch_config, "scales", None),
            patches_per_image=int(_cfg_value(patch_config, "patches_per_image", 1)),
            samples_per_epoch=_cfg_value(patch_config, "samples_per_epoch", None),
        )
        collate_fn = pad_collate
        shuffle = bool(_cfg_value(patch_config, "shuffle", True))
    else:
        dataset = ImageAndPF(data_path, scale)
        collate_fn = pad_collate if batch_size > 1 else None
        shuffle = False
    max_workers = len(os.sched_getaffinity(0))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max_workers,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
