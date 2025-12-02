import torch
from timm.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F

from gaussianimage_rs import inv_sigmoid
from generalizable_model.convnext_unet import ConvNeXtUnet
from generalizable_model.ellipse_process import EllipseProcessKNN
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum


def render(xy, scaling, rotation, color, H, W):
    tile_bounds = (
        (W + 16 - 1) // 16,
        (H + 16 - 1) // 16,
        1,
    )
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        xy, scaling, rotation, H, W, tile_bounds
    )
    opacity = torch.ones_like(rotation)
    out_img = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        color, opacity, H, W, 16, 16
    )
    # out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
    out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    return {"render": out_img}



def sample_operation(sample_points, feature_map, image):
    feature_samples = F.grid_sample(
        feature_map,
        sample_points.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, C, N, P]
    color_samples = F.grid_sample(
        image,
        sample_points.unsqueeze(0),
        mode="bilinear",
        align_corners=False
    )  # [1, 3, N, P]

    feature_samples = feature_samples.squeeze(0).permute(1, 2, 0).contiguous()  # [N, P, C]
    color_samples = color_samples.squeeze(0).permute(1, 2, 0).contiguous()  # [N, P, 3]
    return feature_samples, color_samples


def neighbor_offsets(centers, neighbors):
    offsets = neighbors - centers.unsqueeze(1)
    return offsets.reshape(centers.shape[0], -1)


def scaling_activation(x):
    x = 50 * torch.tanh(0.02 * x)
    return torch.where(
        x < 0,
        0.5 * torch.exp(0.5 * x),
        F.softplus(0.25 * x - 2.5, beta=-2, threshold=4) + 3
    )


class InitNet(nn.Module):
    def __init__(self, kernel_size=3, knn_k=6, neighbor_sample=3):
        super().__init__()

        self.kernel_size = kernel_size
        self.feature_dim = 64
        self.knn_neighbors = max(1, neighbor_sample)
        self.sample_points = self.knn_neighbors + 1  # neighbors + center
        self.adjacent_count = min(3, self.knn_neighbors)

        self.ell_process = EllipseProcessKNN(k=knn_k, sample_neighbors=self.knn_neighbors)

        self.feature_net = ConvNeXtUnet(
            out_channels=self.feature_dim, encoder_name='convnext_base',
            pretrained=True, in_22k=False, in_channels=3, bilinear=False
        )

        

        self.position_field = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

        feature_reduction_in = self.feature_dim * self.sample_points
        self.feature_reduction = nn.Sequential(
            nn.Linear(feature_reduction_in, feature_reduction_in),
            nn.ReLU(),
            nn.Linear(feature_reduction_in, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        color_feature_dim = self.sample_points * 3
        neighbor_feature_dim = self.knn_neighbors * 2
        ell_feature_dim = 4
        self.mlp_dim = ell_feature_dim + neighbor_feature_dim + color_feature_dim + self.feature_dim
        

        self.bc_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.sample_points),
            nn.Softmax(dim=1)
        )  # [N, sample_points]

        neighbors_xy_dim = (self.adjacent_count + 1) * 2
        self.mlp_dim = self.mlp_dim + neighbors_xy_dim
        self.scale_rot_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, 3)
        )  # [N, 3]

        self.color_field = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, 1),
            nn.Sigmoid()
        )  # [N, 3]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image, get_gaussians=False, only_get_pf=False):  # [N, 2], range is [-1, 1]
        feature_map = self.feature_net(image)  # [1, C, H, W]
        B, C, H, W = feature_map.shape
        out_position = self.position_field(feature_map.view(B, C, -1).permute(0, 2, 1))  # [B, H * W, 1]
        out_position = out_position.view(B, H, W)  # [B, H, W]
        if only_get_pf:
            return out_position, image

        e_center, neighbor_pts, e_size, e_angle, neighbor_indices = self.ell_process.process(out_position, self.kernel_size)
        # neighbor_pts: [N, knn_neighbors, 2]

        sample_points = torch.cat([e_center.unsqueeze(1), neighbor_pts], dim=1)  # [N, sample_points, 2]
        map_feature, color_feature = sample_operation(sample_points, feature_map, image)

        reduced_feature = self.feature_reduction(map_feature.view(-1, self.sample_points * self.feature_dim))
        color_feature = color_feature.view(-1, self.sample_points * 3)
        neighbor_feat = neighbor_offsets(e_center, neighbor_pts)  # [N, knn_neighbors * 2]
        ell_feature = torch.cat(
            (e_center, (e_size[:, 0:1] / (e_size[:, 1:2] + 1e-6)), e_angle.unsqueeze(-1)),
            dim=1
        )  # [N, 4]

        mlp_feature = torch.cat((ell_feature, neighbor_feat, color_feature, reduced_feature), dim=1)

        # Coordinate refinement
        bc = self.bc_field(mlp_feature).unsqueeze(-1)  # [N, sample_points, 1]
        xy = torch.sum(bc * sample_points, dim=1)  # [N, 2]
        xy = torch.clamp(xy, -1 + 1e-6, 1 - 1e-6)

        adj_idx = neighbor_indices[:, :self.adjacent_count]
        adj_points = xy[adj_idx]
        neighbors_xy = torch.cat([xy, adj_points.reshape(xy.shape[0], -1)], dim=1).detach()
        mlp_feature = torch.cat([neighbors_xy, mlp_feature], dim=1)

        scale_rot = self.scale_rot_field(mlp_feature)  # [N, 3]
        scaling = scaling_activation(scale_rot[:, :2]) * e_size  # [N, 2]

        inv_e_angle = inv_sigmoid(e_angle.unsqueeze(-1))
        rotation = torch.tanh(scale_rot[:, 2:3]) + inv_e_angle  # [N, 1]
        rotation = torch.sigmoid(rotation)  # [N, 1]

        # TODO simplify color reshaping
        sampled_color = F.grid_sample(
            image,
            xy.unsqueeze(0).unsqueeze(0).detach(),  # [1, 1, N, 2]
            mode="bilinear",
            align_corners=False
        )  # [1, 3, 1, N]
        sampled_color = sampled_color.squeeze(0).squeeze(1).permute(1, 0)  # [N, 3]
        color = self.color_field(mlp_feature) * sampled_color

        if get_gaussians:
            return xy, scaling, rotation, color, neighbor_pts
        else:
            rot_rad = rotation * 2 * torch.pi
            render_img = render(xy, scaling, rot_rad, color, H, W)["render"]

            # import matplotlib.pyplot as plt
            # plt.imshow(render_img[0].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()
            # exit(0)

            return out_position, render_img, scaling
