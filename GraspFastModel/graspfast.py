import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from mle import MLE
from points_graspable import PCS, GCE
from lgd import LGD
from wadnet import CRR, WADNet
from loss_utils import GRASP_MAX_WIDTH, NUM_DIRECTION, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_direction_and_label, batch_directionpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

class GraspFast(nn.Module):
    def __init__(self, cylinder_radius=0.05, feat_channel=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.feat_channel = feat_channel
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.NUM_DIRECTION = NUM_DIRECTION

        self.mle = MLE(in_channels=3, out_channels=self.feat_channel, D=3)
        self.gce = GCE(feat_channel=self.feat_channel)
        self.pcs = PCS(feat_channel=self.feat_channel)
        self.lgd = LGD(self.NUM_DIRECTION, feat_channel=self.feat_channel, is_training=self.is_training)

        self.crr1 = CRR(nsample=16, cylinder_radius=0.025, feat_channel=self.feat_channel)
        self.crr2 = CRR(nsample=16, cylinder_radius=0.05, feat_channel=self.feat_channel)
        self.crr3 = CRR(nsample=16, cylinder_radius=0.075, feat_channel=self.feat_channel)
        self.crr4 = CRR(nsample=16, cylinder_radius=0.1, feat_channel=self.feat_channel)
        self.fuse_multi_scale = nn.Conv1d(256 * 4, 256, 1)
        self.wadnet = WADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']
        B, point_num, _ = seed_xyz.shape
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.mle(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        end_points = self.gce(seed_features, end_points)
        end_points = self.pcs(seed_features, end_points)

        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points['object_criteria']
        graspness_score = end_points['graspable_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_point_inds = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            seed_point_inds.append(fps_idxs.squeeze(0))
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        seed_features_graspable = torch.stack(seed_features_graspable)
        seed_point_inds = torch.stack(seed_point_inds, 0)
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B
        end_points['graspable_mask'] = seed_point_inds

        end_points, res_feat = self.lgd(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_directions_rot, end_points = match_grasp_direction_and_label(end_points)
        else:
            grasp_top_directions_rot = end_points['grasp_top_direction_rot']
        group_features1 = self.crr1(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_directions_rot)
        group_features2 = self.crr2(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_directions_rot)
        group_features3 = self.crr3(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_directions_rot)
        group_features4 = self.crr4(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_directions_rot)
        B, _, M = group_features1.size()
        vp_features_concat = torch.cat([group_features1, group_features2, group_features3, group_features4], dim=1)
        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        end_points = self.wadnet(vp_features_concat, end_points)
        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_direction_xyz'][i].float()
        grasp_rot = batch_directionpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds
