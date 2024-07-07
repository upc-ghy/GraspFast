import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix

# 
class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


# 输入(B, C=512, num_seed=1024)维度的1024个点的特征向量
# 输出(B, num_seed)维度的1024个点的最优抓取方向(有一定的概率采样到其它抓取方向)
class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view           # num_view = 300
        self.in_dim = seed_feature_dim     # seed_feature_dim = 512
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()   # 特征维度应该是(B, C, num_seed), 其中C论文中是512
        res_features = F.relu(self.conv1(seed_features), inplace=True)   # 输出的res_features的维度是(B, C, num_seed)
        features = self.conv2(res_features)     # 输出的结果维度是(B, num_view, num_seed)
        view_score = features.transpose(1, 2).contiguous()  # 转置后的结果维度是：(B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()   # (B, num_seed=1024, num_view=300)
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)  # view_score_的维度是(B, num_seed, num_view)

            top_view_inds = []
            # 因为torch.multinomial的功能限制，因此必须按照Batch进行拆开
            for i in range(B):
                # torch.multinomial(input, num_samples, replacement=False, *, generator=None) 是 PyTorch 中的一个函数，用于从多项式分布中进行采样。它有以下几个参数：
                # input: 输入的概率分布张量，形状为 (N, K)，其中 N 是样本数量，K 是类别数量。每个样本对应一个概率分布，表示从这个分布中进行采样。
                # num_samples: 采样的数量，即要从每个概率分布中采样多少个样本。
                # replacement: 是否可以进行重复采样。如果设置为 True，则采样时可以重复选择同一个样本；如果设置为 False，则采样时不会重复选择同一个样本。
                # generator （可选）：用于生成随机数的随机数生成器。如果不提供，则使用默认的随机数生成器。
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)  # view_score_[i]的维度是(num_seed, num_view)
                top_view_inds.append(top_view_inds_batch)
            # 因为每个采样点只按照多项式概率分布采样了一个分数最高的抓取方向，所以维度由(B, num_seed, num_view)缩减为(b, num_seed)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample  # 16，每个圆柱体范围内采样16个邻域点
        self.in_dim = seed_feature_dim   # 512
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3, mlps=[3+512, 256, 256]

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        # 这个self.grouper的第一项是否够优，不用1024个seed_xyz_graspable的坐标，而使用真个点云的数据呢
        # self.grouper后输出的结果grouped_feature维度是(B, C+3, num_seed, nsample)=(B, 512+3, 1024, 16)
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)=(B, 256, 1024, 16)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()      # 这个vp_features的维度是(B, 256, num_seed)
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)   # 网络输出的结果维度是(B,2*num_angle*num_depth, num_seed)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)  # 变换后的维度为(B, 2, num_angle, num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)    # 变换后的维度为(B, 2, num_seed, num_angle, num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
