import math
import torch
from timm.layers import trunc_normal_
from torch import nn

from generalizable_model.convnext_unet import ConvNeXtUnet
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum


def render(xy_pixels, scaling_pixels, rotation, color, opacity, H, W):
    # Convert pixel-space coordinates/scales back to normalized device coordinates for gsplat
    if W <= 1 or H <= 1:
        raise ValueError("Image width and height must be greater than 1 for rasterization.")
    xy = xy_pixels.clone()
    scaling = scaling_pixels.clone()
    W_f = float(W)
    H_f = float(H)
    xy[:, 0:1] = (xy_pixels[:, 0:1] / (W_f - 1e-4)) * 2.0 - 1.0
    xy[:, 1:2] = (xy_pixels[:, 1:2] / (H_f - 1e-4)) * 2.0 - 1.0
    scaling[:, 0:1] = scaling_pixels[:, 0:1] * (2.0 / W_f)
    scaling[:, 1:2] = scaling_pixels[:, 1:2] * (2.0 / H_f)

    tile_bounds = (
        (W + 16 - 1) // 16,
        (H + 16 - 1) // 16,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy, scaling, rotation, H, W, tile_bounds
    )
    out_img = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        color, opacity, H, W, 16, 16
    )
    out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    return {"render": out_img}


class E2ENet(nn.Module):
    def __init__(
        self,
        sampling_temperature=1.0,
        hard_sampling=True,
        min_scale=5e-3,
        max_scale=0.4,
        default_keep_ratio=0.1,
        backbone_pretrained=True,
        freeze_backbone=False,
        init_gaussian_scale=0.12,
        init_gaussian_opacity=0.6,
    ):
        super().__init__()

        self.feature_dim = 64
        self.temperature = sampling_temperature
        self.hard_sampling = hard_sampling
        self.min_scale = min_scale
        self.max_scale = max_scale
        if self.max_scale <= self.min_scale:
            raise ValueError("max_scale must be greater than min_scale")
        self.scale_range = self.max_scale - self.min_scale
        self.default_keep_ratio = default_keep_ratio
        self.init_gaussian_scale = init_gaussian_scale
        self.init_gaussian_opacity = init_gaussian_opacity
        self.max_offset = 2.0  # allow blobs to move freely across the image
        self.reg_min_scale_px = 3.0
        self.reg_min_opacity = 0.3
        self.reg_min_color = 0.05
        self.reg_eps = 1e-6

        self.feature_net = ConvNeXtUnet(
            out_channels=self.feature_dim,
            encoder_name='convnext_base',
            pretrained=backbone_pretrained,
            in_22k=False,
            in_channels=3,
            bilinear=False
        )
        if freeze_backbone:
            for param in self.feature_net.parameters():
                param.requires_grad = False
            self.feature_net.eval()

        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Importance (LOS) logits
        self.sampling_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, 1, kernel_size=1)
        )

        # Per-pixel Gaussian params: dx, dy, sx, sy, rot, rgb(3), opacity
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, 9, kernel_size=1)
        )

        self.apply(self._init_weights)
        self._init_gaussian_head_bias()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _safe_logit(value, eps=1e-4):
        clamped = min(max(value, eps), 1 - eps)
        return math.log(clamped) - math.log(1 - clamped)

    def _init_gaussian_head_bias(self):
        # Tailor biases for the gaussian head so we start with visible blobs.
        final_layer = None
        for layer in reversed(self.gaussian_head):
            if isinstance(layer, nn.Conv2d) and layer.out_channels == 9:
                final_layer = layer
                break
        if final_layer is None or final_layer.bias is None:
            return
        with torch.no_grad():
            bias = final_layer.bias
            # Offsets start centered.
            bias[0:2].zero_()
            # Target a moderate spatial footprint in normalized coordinates.
            target_scale = min(max(self.init_gaussian_scale, self.min_scale + 1e-4), self.max_scale - 1e-4)
            scale_ratio = (target_scale - self.min_scale) / max(self.scale_range, 1e-4)
            scale_bias = self._safe_logit(scale_ratio)
            bias[2] = scale_bias
            bias[3] = scale_bias
            # Default rotation straight up.
            bias[4] = 0.0
            # Slightly vivid starting color.
            color_bias = self._safe_logit(0.55)
            bias[5:8].fill_(color_bias)
            # Encourage non-zero opacity out of the gate.
            bias[8] = self._safe_logit(self.init_gaussian_opacity)

    def _gaussian_regularizer(self, scaling_pixels, opacity, base_color, mask):
        weights = mask.squeeze(-1)
        denom = weights.sum() + self.reg_eps
        scale_mag = scaling_pixels.mean(dim=-1)
        scale_reg = torch.relu(self.reg_min_scale_px - scale_mag) * weights
        scale_reg = scale_reg.sum() / denom

        opacity_flat = opacity.squeeze(-1)
        opacity_reg = torch.relu(self.reg_min_opacity - opacity_flat) * weights
        opacity_reg = opacity_reg.sum() / denom

        color_mag = base_color.mean(dim=-1)
        color_reg = torch.relu(self.reg_min_color - color_mag) * weights
        color_reg = color_reg.sum() / denom

        total = scale_reg + opacity_reg + color_reg
        return {
            "total": total,
            "scale": scale_reg,
            "opacity": opacity_reg,
            "color": color_reg,
        }

    def _make_grid(self, H, W, device, dtype):
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid

    def _gumbel_topk_mask(self, logits, k):
        B = logits.shape[0]
        flat_logits = logits.view(B, -1)
        total = flat_logits.shape[1]
        if k is None:
            return torch.sigmoid(flat_logits).view_as(logits)

        if isinstance(k, torch.Tensor):
            k_values = k.to(device=flat_logits.device, dtype=torch.long).view(-1)
        elif isinstance(k, (list, tuple)):
            k_values = torch.tensor(k, device=flat_logits.device, dtype=torch.long)
        else:
            k_values = torch.tensor([int(k)], device=flat_logits.device, dtype=torch.long)
        if k_values.numel() < B:
            repeat = (B + k_values.numel() - 1) // k_values.numel()
            k_values = k_values.repeat(repeat)
        k_values = k_values[:B]
        k_values = torch.clamp(k_values, 1, total)

        if self.training:
            noise = torch.rand_like(flat_logits)
            gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
            scores = (flat_logits + gumbel) / self.temperature
        else:
            scores = flat_logits

        hard_mask = torch.zeros_like(flat_logits)
        for b in range(B):
            kb = int(k_values[b].item())
            top_idx = torch.topk(scores[b], k=kb, dim=-1, sorted=False).indices
            hard_mask[b].scatter_(0, top_idx, 1.0)
        soft_mask = torch.sigmoid(flat_logits)
        mask = soft_mask + (hard_mask - soft_mask).detach()
        return mask.view_as(logits)

    def _threshold_mask(self, logits, threshold):
        soft = torch.sigmoid(logits)
        hard = (soft >= threshold).float()
        return soft + (hard - soft).detach()

    def forward(self, image, get_gaussians=False, top_k=None, keep_ratio=None, threshold=None, prune_in_train=False):
        feature_map = self.feature_net(image)
        feature_map = self.pixel_encoder(feature_map)
        B, C, H, W = feature_map.shape

        importance_logits = self.sampling_head(feature_map)

        keep_ratio_eff = keep_ratio
        if keep_ratio_eff is None and top_k is None and threshold is None:
            keep_ratio_eff = self.default_keep_ratio

        target_k = None
        if top_k is not None:
            target_k = top_k
        elif keep_ratio_eff is not None:
            target_k = max(1, int(keep_ratio_eff * H * W))

        if target_k is not None:
            sampling_mask = self._gumbel_topk_mask(importance_logits, target_k)
        elif threshold is not None:
            sampling_mask = self._threshold_mask(importance_logits, threshold)
        else:
            sampling_mask = torch.sigmoid(importance_logits)

        grid = self._make_grid(H, W, image.device, image.dtype)
        pixel_scale = torch.tensor([W / 2.0, H / 2.0], device=image.device, dtype=image.dtype)

        gaussian_params = self.gaussian_head(feature_map)
        params_flat = gaussian_params.permute(0, 2, 3, 1).reshape(B, -1, 9)
        mask_flat = sampling_mask.permute(0, 2, 3, 1).reshape(B, -1, 1)

        xy_offset = torch.tanh(params_flat[..., 0:2]) * self.max_offset
        grid_flat = grid.reshape(-1, 2).unsqueeze(0).expand(B, -1, -1)
        xy = torch.clamp(grid_flat + xy_offset, -1 + 1e-4, 1 - 1e-4)

        scale_logits = params_flat[..., 2:4]
        scaling = torch.sigmoid(scale_logits) * self.scale_range + self.min_scale
        rotation = torch.sigmoid(params_flat[..., 4:5]) * 2 * torch.pi
        base_color = torch.sigmoid(params_flat[..., 5:8])
        raw_opacity = torch.sigmoid(params_flat[..., 8:9])
        scaling_pixels = scaling * pixel_scale
        reg_terms = self._gaussian_regularizer(scaling_pixels, raw_opacity, base_color, mask_flat)

        color = base_color * mask_flat
        opacity = raw_opacity * mask_flat

        xy_pixels = torch.empty_like(xy)
        xy_pixels[..., 0:1] = torch.clamp((xy[..., 0:1] + 1.0) * 0.5 * W, 0.0, W - 1e-4)
        xy_pixels[..., 1:2] = torch.clamp((xy[..., 1:2] + 1.0) * 0.5 * H, 0.0, H - 1e-4)

        render_imgs = []
        xy_all = []
        scaling_all = []
        rotation_all = []
        color_all = []
        opacity_all = []

        prune = (not self.training) or prune_in_train
        for b in range(B):
            xy_b = xy_pixels[b]
            scale_b = scaling_pixels[b]
            rot_b = rotation[b]
            color_b = color[b]
            opacity_b = opacity[b]
            mask_b = mask_flat[b].view(-1)
            logits_b = importance_logits[b].reshape(-1)

            if prune and (top_k is not None or threshold is not None or keep_ratio is not None):
                active_idx = torch.nonzero(mask_b > 0, as_tuple=False).squeeze(-1)
                if active_idx.numel() == 0:
                    fallback_idx = torch.argmax(logits_b)
                    active_idx = fallback_idx.unsqueeze(0)
                xy_b = xy_b[active_idx]
                scale_b = scale_b[active_idx]
                rot_b = rot_b[active_idx]
                color_b = color_b[active_idx]
                opacity_b = opacity_b[active_idx]

            xy_all.append(xy_b)
            scaling_all.append(scale_b)
            rotation_all.append(rot_b)
            color_all.append(color_b)
            opacity_all.append(opacity_b)

            render_img = render(xy_b, scale_b, rot_b, color_b, opacity_b, H, W)["render"]
            render_imgs.append(render_img)

        xy_cat = torch.cat(xy_all, dim=0)
        scaling_cat = torch.cat(scaling_all, dim=0)
        rotation_cat = torch.cat(rotation_all, dim=0)
        color_cat = torch.cat(color_all, dim=0)
        opacity_cat = torch.cat(opacity_all, dim=0)
        render_img = torch.cat(render_imgs, dim=0)

        sampling_field = sampling_mask.squeeze(1)
        if get_gaussians:
            return xy_cat, scaling_cat, rotation_cat, color_cat, opacity_cat, sampling_field

        aux = {
            "gaussian_reg": reg_terms["total"],
            "scale_reg": reg_terms["scale"],
            "opacity_reg": reg_terms["opacity"],
            "color_reg": reg_terms["color"],
        }

        return sampling_field, render_img, aux
