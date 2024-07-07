""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import MinkUNet14D
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation


class GraspFast(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim   # seed_feat_dim=512
        self.num_depth = NUM_DEPTH   # NUM_DEPTH=4
        self.num_angle = NUM_ANGLE   # NUM_ANGLE=12
        self.M_points = M_POINT      # M_POINT=1024
        self.num_view = NUM_VIEW     # NUM_VIEW=300

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        # self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.crop1 = CloudCrop(nsample=16, cylinder_radius=0.025, seed_feature_dim=self.seed_feature_dim)
        self.crop2 = CloudCrop(nsample=16, cylinder_radius=0.05, seed_feature_dim=self.seed_feature_dim)
        self.crop3 = CloudCrop(nsample=16, cylinder_radius=0.075, seed_feature_dim=self.seed_feature_dim)
        self.crop4 = CloudCrop(nsample=16, cylinder_radius=0.1, seed_feature_dim=self.seed_feature_dim)
        self.fuse_multi_scale = nn.Conv1d(256 * 4, 256, 1)
        # self.num_angle=12, self.num_depth=4
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3=B*15000*3，从原始点云中随机采样的是15000个点作为网络输出的原始点云
        B, point_num, _ = seed_xyz.shape  # (B, point_num=15000, 3)
        # point-wise features
        coordinates_batch = end_points['coors']   # 获得放大(1/0.005)倍的坐标值，获得映射到大小为0.005的体素空间中的坐标值张量
        features_batch = end_points['feats']      # 获得维度为(B, point_num, 3)的值全为1的特征张量
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)  # 创建一个Minkowski稀疏张量
        # .F或者.features：代表特征（features），通常用来表示网络前向计算后的输出特征。.C：代表坐标（coordinates），通常用来表示网络前向计算后的输出坐标值。
        seed_features = self.backbone(mink_input).F   # 输出学习到的MinkowskiEngine特征维度是(B,num_point,C=512)
        # end_points['quantize2original']是一个整数数组，其长度为稀疏化坐标数组的长度。它的每个元素保存着一个索引值，用于将稀疏化后的坐标和特征重新映射回原始坐标和特征上。例如，假设稀疏化后的坐标数组 quantized_coordinates_batch 的第 i 个元素是有效的，那么通过索引 quantize2original[i] 可以找到原始坐标数组 coordinates_batch 中等于 quantized_coordinates_batch[i] 的坐标在批次中的位置。这样，就可以将稀疏化后的坐标和特征与原始坐标和特征对应起来，便于后续的计算和处理。
        # 此时的seed_features是个量化后的稀疏矩阵，end_points['quantize2original']中每个元素表示量化后特征表示中的每个值应该对应的原始特征表示中的哪个值。通过这个映射关系，可以将量化后的特征表示转换成原始特征表示，并在后续的操作中使用原始特征表示。
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2) # (2, 1024, 20000)

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']   # 这是self.graspable预测出的关于objectness的两个指标(B, 2, point_num)
        graspness_score = end_points['graspness_score'].squeeze(1)  # 这是self.graspable预测出的关于graspness的一个指标(B, 1, point_num), squeeze(1)之后维度变成(B, point_num)
        # 使用torch.argmax(objectness_score, 1)的作用是在维度为1的轴上找到objectness_score中最大值所在的索引
        objectness_pred = torch.argmax(objectness_score, 1)  # (B, point_num)
        # 是每个点的两个objectness scores分第二个分数大的掩码
        objectness_mask = (objectness_pred == 1)   # 因为只有2个值，两个中的最大值的索引，这里只要第二个值是最大值的情况，那说明第二个值表示是objectness,第一个值表示非objectness的score
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD # 每个点的graspness score高于阈值的掩码
        graspable_mask = objectness_mask & graspness_mask      # 求两个掩码的交集
        # 计算多尺度权重损失时需要使用
        # end_points['graspable_mask'] = graspable_mask  # 这个维度是(B, Ns=20000)

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_point_inds = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]   # 取出一个点云的是否可抓取掩码
            graspable_num_batch += cur_mask.sum()   # 求出一个点云中所有graspable点的数目，并将结果加入到一个batch总可抓点数目中
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim, 通过可抓取掩码，取出一个点云中所有可抓取点特征向量
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3, 通过可抓取掩码，取出一个点云中所有可抓点的坐标值

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)   # (1, M) 返回利用最远点采样获得的M个点的索引
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3，我觉得是(M, 3)
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns, 我觉得是(feat_dim=C, M)

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            seed_point_inds.append(fps_idxs.squeeze(0))
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3， 我觉得是(B, M, 3)
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns, 我觉得是(B, C, M)
        seed_point_inds = torch.stack(seed_point_inds, 0)  # 维度是(B, M)
        end_points['xyz_graspable'] = seed_xyz_graspable   # 把M个可抓取点的xyz坐标存起来
        end_points['graspable_count_stage1'] = graspable_num_batch / B   # 求出每个batch有多少个可抓点
        end_points['graspable_mask'] = seed_point_inds

        # 使用Approach网络，输入学习到的M个seed points的特征，输出学习到的维度为(B, C, num_seed=M=1024)的res_feat特征，以及按照每个seed point的300个接近方向的graspness score(不同于点的graspness score)为概率采样得到的一个视角的索引，维度为(B, num_seed=M)
        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat    # 特征融合后的维度为(B, C, M)

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        # group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        group_features1 = self.crop1(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        group_features2 = self.crop2(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        group_features3 = self.crop3(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        group_features4 = self.crop4(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        
        B, _, M = group_features1.size()
        vp_features_concat = torch.cat([group_features1, group_features2, group_features3, group_features4], dim=1)
        # vp_features_concat = vp_features_concat.view(B, -1, num_seed * num_depth)
        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        # vp_features_concat = vp_features_concat.view(B, -1, num_seed, num_depth)
        end_points = self.swad(vp_features_concat, end_points)
        
        
        # end_points = self.swad(group_features, end_points)

        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds
