import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from loss_utils import generate_grasp_directions, batch_directionpoint_params_to_matrix

class LGD(nn.Module):
    def __init__(self, num_direction, feat_channel, is_training=True):
        super().__init__()
        self.num_direction = num_direction
        self.in_dim = feat_channel
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_direction, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        direction_score = features.transpose(1, 2).contiguous()
        end_points['direction_score'] = direction_score

        if self.is_training:
            direction_score_ = direction_score.clone().detach()
            direction_score_max, _ = torch.max(direction_score_, dim=2)
            direction_score_min, _ = torch.min(direction_score_, dim=2)
            direction_score_max = direction_score_max.unsqueeze(-1).expand(-1, -1, self.num_direction)
            direction_score_min = direction_score_min.unsqueeze(-1).expand(-1, -1, self.num_direction)
            direction_score_ = (direction_score_ - direction_score_min) / (direction_score_max - direction_score_min + 1e-8)

            top_direction_inds = []
            for i in range(B):
                top_direction_inds_batch = torch.multinomial(direction_score_[i], 1, replacement=False)
                top_direction_inds.append(top_direction_inds_batch)
            top_direction_inds = torch.stack(top_direction_inds, dim=0).squeeze(-1)
        else:
            _, top_direction_inds = torch.max(direction_score, dim=2)

            top_direction_inds_ = top_direction_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_directions = generate_grasp_directions(self.num_direction).to(features.device)
            template_directions = template_directions.view(1, 1, self.num_direction, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_directions, 2, top_direction_inds_).squeeze(2)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_directionpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_direction_xyz'] = vp_xyz
            end_points['grasp_top_direction_rot'] = vp_rot

        end_points['grasp_top_direction_inds'] = top_direction_inds
        return end_points, res_features