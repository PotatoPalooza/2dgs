import math
import torch
from timm.layers import trunc_normal_
from torch import nn

from generalizable_model.convnext_unet import ConvNeXtUnet
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum


def render(xy_ndc, scaling_ndc, rotation, color, opacity, H, W):
    """Project gaussians given in normalized device coordinates to rasterized image."""
    if W <= 1 or H <= 1:
        raise ValueError("Image width and height must be greater than 1 for rasterization.")

    tile_bounds = (
        (W + 16 - 1) // 16,
        (H + 16 - 1) // 16,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy_ndc, scaling_ndc, rotation, H, W, tile_bounds
    )
    out_img = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        color, opacity, H, W, 16, 16
    )
    out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    return {"render": out_img}


class ME2ENet(nn.Module):
    """Minimal end-to-end gaussian renderer with per-pixel MLP heads."""

    def __init__(
        self,
        min_scale=5e-3,
        max_scale=0.35,
        max_offset=1.25,
        min_opacity=0.05,
        default_opacity=0.6,
        backbone_pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        if max_scale <= min_scale:
            raise ValueError("max_scale must be greater than min_scale")
        self.feature_dim = 64
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.scale_range = self.max_scale - self.min_scale
        self.max_offset = float(max_offset)
        self.min_opacity = float(min_opacity)
        self.default_opacity = float(default_opacity)
        self.reg_eps = 1e-6

        self.feature_net = ConvNeXtUnet(
            out_channels=self.feature_dim,
            encoder_name="convnext_base",
            pretrained=backbone_pretrained,
            in_22k=False,
            in_channels=3,
            bilinear=False,
        )
        if freeze_backbone:
            for param in self.feature_net.parameters():
                param.requires_grad = False
            self.feature_net.eval()

        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.importance_head = nn.Conv2d(self.feature_dim, 1, kernel_size=1)

        # dx, dy, sx, sy, rot, rgb(3), opacity
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, 9, kernel_size=1),
        )

        self.apply(self._init_weights)
        self._init_gaussian_bias()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_gaussian_bias(self):
        final_conv = None
        for layer in reversed(self.gaussian_head):
            if isinstance(layer, nn.Conv2d) and layer.out_channels == 9:
                final_conv = layer
                break
        if final_conv is None or final_conv.bias is None:
            return
        with torch.no_grad():
            bias = final_conv.bias
            bias[0:2].zero_()  # offsets start centered
            scale_ratio = (0.1 - self.min_scale) / max(self.scale_range, 1e-4)
            scale_bias = math.log(max(scale_ratio, 1e-4)) - math.log(max(1 - scale_ratio, 1e-4))
            bias[2] = scale_bias
            bias[3] = scale_bias
            bias[4] = 0.0  # neutral rotation
            color_bias = math.log(0.55) - math.log(1 - 0.55)
            bias[5:8].fill_(color_bias)
            opacity_bias = math.log(self.default_opacity) - math.log(1 - self.default_opacity)
            bias[8] = opacity_bias

    @staticmethod
    def _make_grid(H, W, device, dtype):
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def _regularize(self, scaling_pixels, opacity, base_color, min_scale_px):
        scale_mag = scaling_pixels.mean(dim=-1)
        scale_reg = torch.relu(min_scale_px - scale_mag)

        opacity_reg = torch.relu(self.min_opacity - opacity.squeeze(-1))
        color_reg = torch.relu(0.05 - base_color.mean(dim=-1))

        total = (scale_reg + opacity_reg + color_reg).mean()
        return {
            "total": total,
            "scale": scale_reg.mean(),
            "opacity": opacity_reg.mean(),
            "color": color_reg.mean(),
        }

    def forward(self, image, get_gaussians=False):
        feature_map = self.feature_net(image)
        feature_map = self.pixel_mlp(feature_map)
        B, _, H, W = feature_map.shape

        importance_logits = self.importance_head(feature_map)
        importance = torch.sigmoid(importance_logits)

        grid = self._make_grid(H, W, image.device, image.dtype)
        pixel_scale = torch.tensor([W / 2.0, H / 2.0], device=image.device, dtype=image.dtype)

        gaussian_params = self.gaussian_head(feature_map)
        params_flat = gaussian_params.permute(0, 2, 3, 1).reshape(B, -1, 9)
        importance_flat = importance.permute(0, 2, 3, 1).reshape(B, -1, 1)

        offset = torch.tanh(params_flat[..., 0:2]) * self.max_offset
        base_grid = grid.reshape(-1, 2).unsqueeze(0).expand(B, -1, -1)
        xy = torch.clamp(base_grid + offset, -1 + 1e-4, 1 - 1e-4)

        scale_logits = params_flat[..., 2:4]
        scaling = torch.sigmoid(scale_logits) * self.scale_range + self.min_scale
        rotation = torch.sigmoid(params_flat[..., 4:5]) * 2 * torch.pi
        base_color = torch.sigmoid(params_flat[..., 5:8])
        opacity_base = torch.sigmoid(params_flat[..., 8:9])
        opacity = torch.clamp(opacity_base, min=self.min_opacity, max=1.0)

        scaling_pixels = scaling * pixel_scale
        min_scale_px = self.min_scale * pixel_scale.mean()
        reg_terms = self._regularize(scaling_pixels, opacity, base_color, min_scale_px)

        color = base_color * importance_flat
        opacity = opacity * importance_flat

        render_imgs = []
        xy_all = []
        scaling_all = []
        rotation_all = []
        color_all = []
        opacity_all = []

        for b in range(B):
            xy_b = xy[b]
            scale_b = scaling[b]
            rot_b = rotation[b]
            color_b = color[b]
            opacity_b = opacity[b]

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

        importance_field = importance.squeeze(1)
        if get_gaussians:
            xy_pixels = xy_cat.clone()
            xy_pixels[..., 0:1] = torch.clamp((xy_cat[..., 0:1] + 1.0) * 0.5 * W, 0.0, W - 1e-4)
            xy_pixels[..., 1:2] = torch.clamp((xy_cat[..., 1:2] + 1.0) * 0.5 * H, 0.0, H - 1e-4)
            scaling_pixels_cat = scaling_cat * pixel_scale
            return xy_pixels, scaling_pixels_cat, rotation_cat, color_cat, opacity_cat, importance_field

        aux = {
            "gaussian_reg": reg_terms["total"],
            "scale_reg": reg_terms["scale"],
            "opacity_reg": reg_terms["opacity"],
            "color_reg": reg_terms["color"],
        }

        return importance_field, render_img, aux
