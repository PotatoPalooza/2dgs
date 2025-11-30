import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
import wandb

from generalizable_model.datasets import get_data_loader
from generalizable_model.me2e_net import ME2ENet
from generalizable_model.utils import RGBLoss
from utils import compute_psnr, Averager


def _namespace_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    return obj


class Trainer:
    def __init__(self, args):
        patch_cfg = _namespace_to_dict(getattr(args.dataset, "patch_sampling", None))
        train_patch_cfg = self._resolve_patch_cfg(patch_cfg, "train")
        valid_patch_cfg = self._resolve_patch_cfg(patch_cfg, "valid")

        self.train_dataloader = get_data_loader(
            args.dataset.train_dir, args.dataset.batch_size, scale=args.dataset.scale, patch_config=train_patch_cfg
        )
        self.valid_dataloader = get_data_loader(
            args.dataset.valid_dir, args.dataset.batch_size, scale=args.dataset.scale, patch_config=valid_patch_cfg
        )

        model_kwargs = {
            "min_scale": float(getattr(args.model, "min_scale", 5e-3)),
            "max_scale": float(getattr(args.model, "max_scale", 0.35)),
            "max_offset": float(getattr(args.model, "max_offset", 1.25)),
            "min_opacity": float(getattr(args.model, "min_opacity", 0.05)),
            "default_opacity": float(getattr(args.model, "default_opacity", 0.6)),
            "backbone_pretrained": bool(getattr(args.model, "backbone_pretrained", True)),
            "freeze_backbone": bool(getattr(args.model, "freeze_backbone", False)),
        }
        self.model = ME2ENet(**model_kwargs).cuda()

        self.epochs = int(getattr(args.train, "epochs", 0))
        self.lr_max = float(args.train.lr)
        self.lr_min = float(args.train.lr_min)
        self.gaussian_reg_weight = float(getattr(args.train, "gaussian_reg_weight", 0.05))
        grad_clip = getattr(args.train, "grad_clip_norm", None)
        self.grad_clip_norm = float(grad_clip) if grad_clip is not None else None
        self.log_interval = int(getattr(args.train, "log_interval", 10))
        self.use_wandb = bool(getattr(args.train, "use_wandb", True))

        self.rgb_loss_fn = RGBLoss(lambda_val=0.7)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr_min
        )
        self.checkpoints_dir = args.train.checkpoints_dir
        self.best_psnr = 0
        self.start_epoch = 0
        self.args = args
        self.run_name = None

        if args.train.resume:
            checkpoint = torch.load(args.train.resume_checkpoint)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_psnr = checkpoint.get("best_psnr", 0)
            print(f"Resume from {args.train.resume_checkpoint}, start from epoch {self.start_epoch}")

    @staticmethod
    def _resolve_patch_cfg(cfg_dict, split):
        if cfg_dict is None:
            return None
        if isinstance(cfg_dict, dict) and any(k in cfg_dict for k in ("train", "valid")):
            return cfg_dict.get(split)
        return cfg_dict

    def _prepare_run(self):
        if self.use_wandb:
            wandb.init(project="Instant-GI", tags=["train_me2e"], config=_namespace_to_dict(self.args))
            self.run_name = wandb.run.name
        if self.run_name is not None:
            self.checkpoints_dir = f"{self.checkpoints_dir}_{self.run_name}"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_epoch(self, epoch_dict):
        epoch_id = epoch_dict["epoch"]
        save_dict = {
            "epoch": epoch_id,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loss": epoch_dict["train_loss"],
            "best_psnr": self.best_psnr,
        }
        if epoch_id % 5 == 0:
            torch.save(save_dict, os.path.join(self.checkpoints_dir, f"epoch_{epoch_id}.pth"))
        torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_last.pth"))
        if "valid_psnr" in epoch_dict and epoch_dict["valid_psnr"] > self.best_psnr:
            self.best_psnr = epoch_dict["valid_psnr"]
            torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_best.pth"))

    def train(self):
        self._prepare_run()
        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_epoch(epoch)
            valid_stats = self.validate(epoch)
            epoch_record = {
                "epoch": epoch,
                **train_stats,
                **valid_stats,
            }
            self.save_epoch(epoch_record)
            self.scheduler.step()
            if self.use_wandb:
                wandb.log({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)
        print("Training finished.")
        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, epoch):
        avg_loss = Averager()
        avg_rgb_loss = Averager()
        avg_gauss_reg_loss = Averager()
        avg_psnr = Averager()

        num_batches = len(self.train_dataloader)
        pbar = tqdm(total=num_batches)

        self.model.train()
        vis_importance = []
        vis_render = []
        for i, (gt_image, _) in enumerate(self.train_dataloader):
            global_step = epoch * num_batches + i
            self.optimizer.zero_grad()
            gt_image = gt_image.cuda()

            importance_map, render_img, aux = self.model(gt_image)

            weight = torch.ones_like(importance_map)
            rgb_loss = self.rgb_loss_fn(render_img, gt_image, weight)
            gauss_reg_loss = aux["gaussian_reg"]

            loss = rgb_loss + self.gaussian_reg_weight * gauss_reg_loss
            loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                avg_loss.add(loss.item())
                avg_rgb_loss.add(rgb_loss.item())
                avg_gauss_reg_loss.add(gauss_reg_loss.item())
                psnr = compute_psnr(render_img, gt_image)
                avg_psnr.add(psnr.item())
                if i % self.log_interval == 0 or i == num_batches - 1:
                    pbar.set_postfix_str(
                        f"Epoch {epoch}, Loss {avg_loss.item():.4f}, "
                        f"rgb {avg_rgb_loss.item():.4f}, reg {avg_gauss_reg_loss.item():.4f}, "
                        f"psnr {avg_psnr.item():.4f}"
                    )
                pbar.update(1)
                if i % max(1, num_batches // 8) == 0:
                    vis_importance.append(importance_map[0])
                    vis_render.append(render_img[0].clamp(0, 1))
                if self.use_wandb and i % self.log_interval == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/rgb_loss": rgb_loss.item(),
                            "train/gaussian_reg": gauss_reg_loss.item(),
                            "train/psnr": psnr.item(),
                        },
                        step=global_step,
                    )
        pbar.close()
        if self.use_wandb:
            wandb.log(
                {
                    "train/importance_map": [wandb.Image(img.detach().cpu().unsqueeze(0)) for img in vis_importance],
                    "train/render": [wandb.Image(img) for img in vis_render],
                },
                step=epoch,
            )
        return {
            "train_loss": avg_loss.item(),
            "train_rgb_loss": avg_rgb_loss.item(),
            "train_gauss_reg": avg_gauss_reg_loss.item(),
            "train_psnr": avg_psnr.item(),
        }

    @torch.no_grad()
    def validate(self, epoch):
        avg_loss = Averager()
        avg_rgb_loss = Averager()
        avg_gauss_reg_loss = Averager()
        avg_psnr = Averager()

        self.model.eval()
        vis_importance = []
        vis_render = []
        pbar = tqdm(total=len(self.valid_dataloader))
        for i, (gt_image, _) in enumerate(self.valid_dataloader):
            gt_image = gt_image.cuda()
            importance_map, render_img, aux = self.model(gt_image)
            weight = torch.ones_like(importance_map)
            rgb_loss = self.rgb_loss_fn(render_img, gt_image, weight)
            gauss_reg_loss = aux["gaussian_reg"]

            loss = rgb_loss + self.gaussian_reg_weight * gauss_reg_loss
            avg_loss.add(loss.item())
            avg_rgb_loss.add(rgb_loss.item())
            avg_gauss_reg_loss.add(gauss_reg_loss.item())
            psnr = compute_psnr(render_img, gt_image)
            avg_psnr.add(psnr.item())
            if i % self.log_interval == 0:
                pbar.set_postfix_str(f"loss {avg_loss.item():.4f}, psnr {avg_psnr.item():.4f}")
            pbar.update(1)
            if i % max(1, len(self.valid_dataloader) // 8) == 0:
                vis_importance.append(importance_map[0])
                vis_render.append(render_img[0].clamp(0, 1))
        pbar.close()
        if self.use_wandb:
            wandb.log(
                {
                    "valid/loss": avg_loss.item(),
                    "valid/rgb_loss": avg_rgb_loss.item(),
                    "valid/gaussian_reg": avg_gauss_reg_loss.item(),
                    "valid/psnr": avg_psnr.item(),
                    "valid/importance_map": [wandb.Image(img.detach().cpu().unsqueeze(0)) for img in vis_importance],
                    "valid/render": [wandb.Image(img) for img in vis_render],
                },
                step=epoch,
            )
        else:
            # Save a small sample grid locally for quick inspection when wandb is disabled.
            sample_dir = os.path.join(self.checkpoints_dir, f"epoch_{epoch:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            for idx, (imp, rend) in enumerate(zip(vis_importance, vis_render)):
                save_image(imp.detach().cpu().unsqueeze(0).clamp(0, 1), os.path.join(sample_dir, f"valid_{idx:03d}_imp.png"))
                save_image(rend.detach().cpu().clamp(0, 1), os.path.join(sample_dir, f"valid_{idx:03d}_render.png"))
        print(
            f"Validation Epoch {epoch}, Loss: {avg_loss.item():.4f}, "
            f"RGB Loss: {avg_rgb_loss.item():.4f}, "
            f"Gauss Reg: {avg_gauss_reg_loss.item():.4f}, "
            f"PSNR: {avg_psnr.item():.4f}"
        )
        return {
            "valid_loss": avg_loss.item(),
            "valid_rgb_loss": avg_rgb_loss.item(),
            "valid_gauss_reg": avg_gauss_reg_loss.item(),
            "valid_psnr": avg_psnr.item(),
        }


def random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script for ME2E.")
    parser.add_argument(
        "--config", type=str, default="./datasets/div2k_train_me2e.yaml", help="Training config path"
    )
    args = parser.parse_args(argv)
    with open(args.config, "r") as f:
        args = yaml.safe_load(f)

    def dict_to_namespace(d):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
        })

    return dict_to_namespace(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    random_seed(getattr(args.train, "seed", None))
    trainer = Trainer(args)
    trainer.train()
