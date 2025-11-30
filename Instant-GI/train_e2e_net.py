import argparse
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml
import os
import wandb
from tqdm import tqdm
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_
from generalizable_model.datasets import get_data_loader
from generalizable_model.e2e_net import E2ENet
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
        # load data
        self.train_dataloader = get_data_loader(
            args.dataset.train_dir, args.dataset.batch_size, scale=args.dataset.scale, patch_config=train_patch_cfg
        )
        self.valid_dataloader = get_data_loader(
            args.dataset.valid_dir, args.dataset.batch_size, scale=args.dataset.scale, patch_config=valid_patch_cfg
        )
        train_len = len(self.train_dataloader)
        valid_len = len(self.valid_dataloader)
        # random select 16 images to visualize
        self.vis_train_indices = sorted(random.sample(range(train_len), min(16, train_len)))
        self.vis_valid_indices = sorted(random.sample(range(valid_len), min(16, valid_len)))

        max_gaussian_scale = float(getattr(args.train, "max_gaussian_scale", 0.4))
        # load model
        self.model = E2ENet(max_scale=max_gaussian_scale).cuda()
        if args.train.pretrained:
            checkpoint = torch.load(args.train.pretrained_checkpoint)
            self.model.load_state_dict(checkpoint["model"])
            print(f"Load model from {args.train.pretrained_checkpoint}")

        self.epochs = args.train.epochs
        self.lr_max = args.train.lr
        self.lr_min = args.train.lr_min

        self.rgb_loss_fn = RGBLoss(lambda_val=0.7)

        self.gaussian_reg_weight = float(getattr(args.train, "gaussian_reg_weight", 0.05))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.train.lr)
        # self.optimizer = Adan(self.model.parameters(), lr=args.train.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr_min
        )
        self.checkpoints_dir = None

        self.args = args
        self.best_psnr = 0
        self.train_keep_ratio = float(getattr(args.train, "train_keep_ratio", 0.1))
        self.infer_keep_ratio = float(getattr(args.train, "infer_keep_ratio", self.train_keep_ratio))
        self.infer_top_k = getattr(args.train, "infer_top_k", None)
        self.infer_threshold = getattr(args.train, "infer_threshold", None)
        grad_clip = getattr(args.train, "grad_clip_norm", None)
        self.grad_clip_norm = float(grad_clip) if grad_clip is not None else None
        self.debug_config = self._build_debug_config(args)
        self.debug_enabled = self.debug_config.enabled
        self.debug_output_dir = None
        self.debug_val_dir = None

        if args.train.resume:
            checkpoint = torch.load(args.train.resume_checkpoint)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_psnr = checkpoint["best_psnr"]
            print(f"Resume from {args.train.resume_checkpoint}, start from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0

    @staticmethod
    def _resolve_patch_cfg(cfg_dict, split):
        if cfg_dict is None:
            return None
        if isinstance(cfg_dict, dict) and any(k in cfg_dict for k in ("train", "valid")):
            return cfg_dict.get(split)
        return cfg_dict


    def save_epoch(self, epoch_dict):
        epoch_id = epoch_dict["epoch"]
        save_dict = {
            "epoch": epoch_id,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loss": epoch_dict["train_loss"],
            "best_psnr": self.best_psnr
        }

        if epoch_id % 4 == 0:
            torch.save(save_dict, os.path.join(self.checkpoints_dir, f"epoch_{epoch_id}.pth"))
        # save last epoch: model, optimizer, epoch_id
        torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_last.pth"))
        if "test_psnr" not in epoch_dict:
            return
        if epoch_dict["test_psnr"] > self.best_psnr:
            self.best_psnr = epoch_dict["test_psnr"]
            torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_best.pth"))

    def train(self):
        # init wandb
        wandb.init(
            project="Instant-GI",
            tags=["train_init"], config=self.args
        )
        run_name = wandb.run.name
        # checkpoints save path
        if run_name is not None:
            self.checkpoints_dir = args.train.checkpoints_dir + "_" + run_name
        else:
            self.checkpoints_dir = args.train.checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.debug_enabled:
            self._prepare_debug_dirs()
            print(
                "Debugging enabled: gradient thresholds (vanish={:.2e}, explode={:.2e}), "
                "gaussian stats interval={}, validation images per epoch={}".format(
                    self.debug_config.grad_vanish_threshold,
                    self.debug_config.grad_explode_threshold,
                    self.debug_config.gaussian_stats_interval,
                    self.debug_config.validation_image_count
                )
            )

        for epoch in range(self.start_epoch, self.epochs):
            epoch_dict = self.train_epoch(epoch)
            if epoch_dict is not None:
                self.save_epoch(epoch_dict)
            self._save_debug_validation_images(epoch)
            self.scheduler.step()
            wandb.log({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)
        print("Training finished.")
        wandb.finish()

    def train_epoch(self, epoch):
        avg_loss = Averager()
        avg_rgb_loss = Averager()
        avg_gauss_reg_loss = Averager()
        avg_psnr = Averager()

        num_batches = len(self.train_dataloader)
        pbar = tqdm(total=num_batches)

        self.model.train()
        train_vis_pred_pfs = []
        train_vis_pred_image = []
        for i, (gt_image, gt_pf) in enumerate(self.train_dataloader):
            global_step = epoch * num_batches + i
            self.optimizer.zero_grad()
            gt_image = gt_image.cuda()
            gt_pf = gt_pf.cuda()  # kept for visualization parity

            pred_pf, render_img, aux = self.model(gt_image, keep_ratio=self.train_keep_ratio, prune_in_train=False)

            rgb_loss_weight = torch.ones_like(pred_pf)
            target_image = gt_image
            rgb_loss = self.rgb_loss_fn(render_img, target_image, rgb_loss_weight)
            gauss_reg_loss = aux["gaussian_reg"]

            loss = rgb_loss + self.gaussian_reg_weight * gauss_reg_loss
            loss.backward()
            self._apply_gradient_clipping(epoch, i, global_step)
            self._check_gradients(epoch, i, global_step)
            self.optimizer.step()
            self._maybe_log_gaussian_stats(gt_image.detach(), epoch, i, global_step)
            with torch.no_grad():
                avg_loss.add(loss.item())
                avg_rgb_loss.add(rgb_loss.item())
                avg_gauss_reg_loss.add(gauss_reg_loss.item())
                psnr = compute_psnr(render_img, gt_image)
                avg_psnr.add(psnr.item())

                if i % 10 == 0 or i == len(self.train_dataloader) - 1:
                    pbar.set_postfix_str(
                        f"Epoch {epoch}, Loss {avg_loss.item():.4f}, "
                        f"lr {self.optimizer.param_groups[0]['lr']:.6f}, "
                        f"rgb_loss {avg_rgb_loss.item():.4f}, "
                        f"gauss_reg {avg_gauss_reg_loss.item():.4f}, "
                        f"psnr {avg_psnr.item():.4f}"
                    )
                pbar.update(1)
                if i in self.vis_train_indices:
                    train_vis_pred_pfs.append(pred_pf[0])
                    train_vis_pred_image.append(render_img[0].clamp(0, 1))
        pbar.close()

        wandb.log({"train/pred_pf": [wandb.Image(img.detach().cpu().clamp(0, 1).unsqueeze(0)) for img in train_vis_pred_pfs]}, step=epoch)
        wandb.log({"train/render_image": [wandb.Image(img.clamp(0, 1)) for img in train_vis_pred_image]}, step=epoch)

        pbar.close()
        wandb.log(
            {"train/loss": avg_loss.item(),
             "train/rgb_loss": avg_rgb_loss.item(),
             "train/gaussian_reg_loss": avg_gauss_reg_loss.item(),
             "train/psnr": avg_psnr.item()},
            step=epoch
        )
        print(
            f"Epoch {epoch}, Loss: {avg_loss.item():.4f}, "
            f"RGB Loss: {avg_rgb_loss.item():.4f}, "
            f"Gauss Reg: {avg_gauss_reg_loss.item():.4f}, "
            f"PSNR: {avg_psnr.item():.4f}"
        )
        if epoch % 2 == 0:
            test_dict = self.test(epoch)
            return {
                "epoch": epoch,
                "train_loss": avg_loss.item(), "train_psnr": avg_psnr.item(),
                "test_loss": test_dict["test_loss"], "test_psnr": test_dict["test_psnr"]
            }
        else:
            return {
                "epoch": epoch,
                "train_loss": avg_loss.item(), "train_psnr": avg_psnr.item()
            }

    def _build_debug_config(self, args):
        debug_cfg = getattr(args.train, "debug", SimpleNamespace())
        return SimpleNamespace(
            enabled=bool(getattr(debug_cfg, "enabled", False)),
            gaussian_stats_interval=int(getattr(debug_cfg, "gaussian_stats_interval", 50)),
            grad_explode_threshold=float(getattr(debug_cfg, "grad_explode_threshold", 1e3)),
            grad_vanish_threshold=float(getattr(debug_cfg, "grad_vanish_threshold", 1e-5)),
            save_validation_images=bool(getattr(debug_cfg, "save_validation_images", True)),
            validation_image_count=int(getattr(debug_cfg, "validation_image_count", 8))
        )

    def _prepare_debug_dirs(self):
        if not self.debug_enabled:
            return
        self.debug_output_dir = os.path.join(self.checkpoints_dir, "debug")
        self.debug_val_dir = os.path.join(self.debug_output_dir, "validation_images")
        os.makedirs(self.debug_output_dir, exist_ok=True)
        os.makedirs(self.debug_val_dir, exist_ok=True)

    def _tensor_stats(self, tensor):
        if tensor is None or tensor.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        flat = tensor.detach().float().view(-1)
        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item())
        }

    def _maybe_log_gaussian_stats(self, gt_image, epoch, batch_idx, global_step):
        if not self.debug_enabled:
            return
        interval = max(1, self.debug_config.gaussian_stats_interval)
        if batch_idx % interval != 0:
            return
        with torch.no_grad():
            _, scaling, rotation, color, opacity, sampling_mask = self.model(
                gt_image,
                get_gaussians=True,
                keep_ratio=self.train_keep_ratio,
                prune_in_train=False
            )
        scale_stats = self._tensor_stats(scaling)
        rot_stats = self._tensor_stats(rotation)
        color_stats = self._tensor_stats(color)
        opacity_stats = self._tensor_stats(opacity)

        r_stats = self._tensor_stats(color[..., 0])
        g_stats = self._tensor_stats(color[..., 1])
        b_stats = self._tensor_stats(color[..., 2])

        mask_stats = self._tensor_stats(sampling_mask)
        stats_msg = (
            f"[Debug][Epoch {epoch} Step {batch_idx}] "
            f"Scale mean {scale_stats['mean']:.4f} std {scale_stats['std']:.4f} min {scale_stats['min']:.4f} max {scale_stats['max']:.4f} | "
            f"Rot mean {rot_stats['mean']:.4f} std {rot_stats['std']:.4f} | "
            f"Color mean {color_stats['mean']:.4f} std {color_stats['std']:.4f} | "
            f"Opacity mean {opacity_stats['mean']:.4f} std {opacity_stats['std']:.4f} | "
            f"R mean {r_stats['mean']:.4f} std {r_stats['std']:.4f} | "
            f"G mean {g_stats['mean']:.4f} std {g_stats['std']:.4f} | "
            f"B mean {b_stats['mean']:.4f} std {b_stats['std']:.4f} | "
            f"Mask mean {mask_stats['mean']:.4f}"
        )
        print(stats_msg)
        wandb.log({
            "debug/scale_mean": scale_stats["mean"],
            "debug/scale_std": scale_stats["std"],
            "debug/rotation_mean": rot_stats["mean"],
            "debug/rotation_std": rot_stats["std"],
            "debug/color_mean": color_stats["mean"],
            "debug/color_std": color_stats["std"],
            "debug/opacity_mean": opacity_stats["mean"],
            "debug/opacity_std": opacity_stats["std"],
            "debug/mask_mean": mask_stats["mean"]
        }, step=global_step)

    def _apply_gradient_clipping(self, epoch, batch_idx, global_step):
        if self.grad_clip_norm is None:
            return
        total_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        if self.debug_enabled:
            wandb.log({"debug/preclip_grad_norm": float(total_norm)}, step=global_step)
            if total_norm > self.grad_clip_norm:
                print(
                    f"[Debug][Epoch {epoch} Step {batch_idx}] Clipped gradient norm "
                    f"from {total_norm:.4e} to <= {self.grad_clip_norm:.4e}"
                )

    def _check_gradients(self, epoch, batch_idx, global_step):
        if not self.debug_enabled:
            return
        total_norm_sq = 0.0
        grad_count = 0
        has_inf = False
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                has_inf = True
                break
            total_norm_sq += grad.norm(2).item() ** 2
            grad_count += 1
        if grad_count == 0:
            return
        if has_inf:
            print(f"[Debug][Epoch {epoch} Step {batch_idx}] Non-finite gradients detected!")
            wandb.log({"debug/grad_issue": 1}, step=global_step)
            return
        grad_norm = total_norm_sq ** 0.5
        log_dict = {"debug/grad_norm": grad_norm}
        warnings = []
        if grad_norm < self.debug_config.grad_vanish_threshold:
            warnings.append("vanishing gradient")
        if grad_norm > self.debug_config.grad_explode_threshold:
            warnings.append("exploding gradient")
        for warn in warnings:
            print(f"[Debug][Epoch {epoch} Step {batch_idx}] Warning: {warn} (norm={grad_norm:.4e})")
        wandb.log(log_dict, step=global_step)

    @torch.no_grad()
    def _save_debug_validation_images(self, epoch):
        if not self.debug_enabled or not self.debug_config.save_validation_images:
            return
        if self.debug_val_dir is None:
            return
        was_training = self.model.training
        self.model.eval()
        target_images = self.debug_config.validation_image_count
        saved = 0
        epoch_dir = os.path.join(self.debug_val_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        for batch_idx, (gt_image, _) in enumerate(self.valid_dataloader):
            gt_image = gt_image.cuda()
            sampling_field, render_img, _ = self.model(
                gt_image,
                top_k=self.infer_top_k,
                keep_ratio=self.infer_keep_ratio,
                threshold=self.infer_threshold
            )
            batch_size = render_img.shape[0]
            for b in range(batch_size):
                if saved >= target_images:
                    break
                sample_name = f"sample_{saved:03d}"
                save_image(
                    render_img[b].detach().cpu().clamp(0, 1),
                    os.path.join(epoch_dir, f"{sample_name}_render.png")
                )
                save_image(
                    gt_image[b].detach().cpu().clamp(0, 1),
                    os.path.join(epoch_dir, f"{sample_name}_gt.png")
                )
                save_image(
                    sampling_field[b].detach().cpu().unsqueeze(0).clamp(0, 1),
                    os.path.join(epoch_dir, f"{sample_name}_sampling_field.png")
                )
                saved += 1
            if saved >= target_images:
                break
        self.model.train(was_training)

    @torch.no_grad()
    def test(self, epoch=0):
        avg_loss = Averager()
        avg_rgb_loss = Averager()
        avg_gauss_reg_loss = Averager()
        avg_psnr = Averager()
        self.model.eval()
        test_vis_pred_pfs = []
        test_vis_pred_image = []

        pbar = tqdm(total=len(self.valid_dataloader))

        for i, (gt_image, gt_pf) in enumerate(self.valid_dataloader):
            gt_image = gt_image.cuda()
            gt_pf = gt_pf.cuda()
            pred_pf, render_img, aux = self.model(
                gt_image,
                top_k=self.infer_top_k,
                keep_ratio=self.infer_keep_ratio,
                threshold=self.infer_threshold
            )
            rgb_loss_weight = torch.ones_like(pred_pf)
            rgb_loss = self.rgb_loss_fn(render_img, gt_image, rgb_loss_weight)
            gauss_reg_loss = aux["gaussian_reg"]
            loss = rgb_loss + self.gaussian_reg_weight * gauss_reg_loss
            avg_loss.add(loss.item())
            avg_rgb_loss.add(rgb_loss.item())
            avg_gauss_reg_loss.add(gauss_reg_loss.item())
            psnr = compute_psnr(render_img, gt_image)
            avg_psnr.add(psnr.item())
            if i % 10 == 0:
                pbar.set_postfix_str(f"loss {avg_loss.item():.4f}, psnr {avg_psnr.item():.4f}")
            pbar.update(1)
            if i in self.vis_valid_indices:
                test_vis_pred_pfs.append(pred_pf[0])
                test_vis_pred_image.append(render_img[0].clamp(0, 1))
        pbar.close()
        wandb.log({"test/pred_pf": [wandb.Image(img.detach().cpu().clamp(0, 1).unsqueeze(0)) for img in test_vis_pred_pfs]}, step=epoch)
        wandb.log({"test/render_image": [wandb.Image(img.clamp(0, 1)) for img in test_vis_pred_image]}, step=epoch)

        pbar.close()
        wandb.log(
            {"test/loss": avg_loss.item(),
             "test/rgb_loss": avg_rgb_loss.item(),
             "test/gaussian_reg_loss": avg_gauss_reg_loss.item(),
             "test/psnr": avg_psnr.item()},
            step=epoch
        )
        print(
            f"Test Loss: {avg_loss.item():.4f}, "
            f"RGB Loss: {avg_rgb_loss.item():.4f}, "
            f"Gauss Reg: {avg_gauss_reg_loss.item():.4f}, "
            f"PSNR: {avg_psnr.item():.4f}"
        )
        return {
            "test_loss": avg_loss.item(), "test_psnr": avg_psnr.item()
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
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--config", type=str, default='./datasets/div2k_train_init_net.yaml', help="Training or testing config"
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
    random_seed(args.train.seed)

    trainer = Trainer(args)
    trainer.train()
