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

class CRR(nn.Module):
    def __init__(self, nsample, feat_channel, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = feat_channel
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)
        new_features = self.mlps(grouped_feature)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        new_features = new_features.squeeze(-1)
        return new_features


class WADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        end_points['grasp_score_pred'] = vp_features[:, 0]
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
