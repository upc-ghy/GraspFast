'''
dsn in uois 3d
Author: Haoxiang Ma
Date: 2021.4.24
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone import Pointnet2Backbone
from pct_zh import PointTransformerBackbone_light, PointTransformerBackbone_lightseg
from pointnet2_utils import three_nn,three_interpolate
import segmentation_loss as ls


class DSN(nn.Module):
    def __init__(self, input_feature_dim=0, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone = PointTransformerBackbone_lightseg()
        self.foreground_module = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 2, 1),
        )
        self.center_direction_module = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud,end_points)
        foreground_logits = self.foreground_module(seed_features)
        center_offsets = self.center_direction_module(seed_features)

        dist, idx = three_nn(pointcloud, seed_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        foreground_logits = three_interpolate(
            foreground_logits, idx, weight
        )

        center_offsets = three_interpolate(
            center_offsets, idx, weight
        )

        end_points["foreground_logits"] = foreground_logits
        end_points["center_offsets"] = center_offsets
        return end_points


def construct_M_logits(predicted_centers, object_centers):
    distances = torch.norm(predicted_centers.unsqueeze(0) - object_centers.unsqueeze(-1), dim=1)

    return -15. * distances


def cluster(xyz_img, offsets, fg_mask):
    clustered_img = torch.zeros_like(fg_mask, dtype=torch.long)
    if torch.sum(fg_mask) == 0:
        return clustered_img, torch.zeros((0, 3), device="cuda")

    predicted_centers = xyz_img + offsets
    predicted_centers = predicted_centers
    ms = ls.GaussianMeanShift(
        max_iters=10,
        epsilon=0.05,
        sigma=0.02,
        num_seeds=50,
        subsample_factor=5
    )
    cluster_labels = ms.mean_shift_smart_init(predicted_centers[fg_mask==1])
    clustered_img[fg_mask==1] = cluster_labels + 1

    uniq_cluster_centers = ms.uniq_cluster_centers
    uniq_labels = ms.uniq_labels + 1

    uniq_counts = torch.zeros_like(uniq_labels)
    for j, label in enumerate(uniq_labels):
        uniq_counts[j] = torch.sum(clustered_img == label)
    valid_indices = []
    for j, label in enumerate(uniq_labels):
        if uniq_counts[j] <10:
            continue
        valid_indices.append(j)
    valid_indices = np.array(valid_indices)

    new_cl_img = torch.zeros_like(clustered_img)
    if valid_indices.shape[0] > 0:
        uniq_cluster_centers = uniq_cluster_centers[valid_indices, :]

        new_label = 1
        for j in valid_indices:
            new_cl_img[clustered_img == uniq_labels[j]] = new_label
            new_label += 1

    else:
        uniq_cluster_centers = torch.zeros((0, 3), dtype=torch.float, device="cuda")
    clustered_img = new_cl_img

    return clustered_img, uniq_cluster_centers


def smart_random_sample_indices(X, Y, num_seeds):
    unique_obj_labels = torch.unique(Y)
    num_objects = unique_obj_labels.shape[0]

    indices = torch.zeros(0, dtype=torch.long, device=X.device)

    num_seeds_per_obj = int(np.ceil(num_seeds / num_objects))
    for k in unique_obj_labels:
        label_indices = torch.where(Y == k)[0]
        randperm = torch.randperm(label_indices.shape[0])
        inds = label_indices[randperm[:num_seeds_per_obj]]
        indices = torch.cat([indices, inds], dim=0)

    X_I = X[indices, :]
    Y_I = Y[indices]

    return X_I, Y_I


def hill_climb_one_iter(Z, X, sigmas):
    W = ls.gaussian_kernel(Z, X, sigmas)
    Q = W / W.sum(dim=1, keepdim=True)
    Z = torch.mm(Q, X)

    return Z


def get_seg_loss(end_points):
    foreground_loss = ls.CELossWeighted(weighted=True)
    center_offset_loss = ls.SmoothL1LossWeighted(weighted=True)
    foreground_labels = end_points['foreground_mask']

    instance_labels = end_points['instance_mask']
    center_offset_labels = end_points['3D_offsets'].permute(0,2,1)

    fg_logits = end_points["foreground_logits"]
    center_offsets = end_points["center_offsets"]
    fg_loss = foreground_loss(fg_logits, foreground_labels)
    center_loss = center_offset_loss(center_offsets, center_offset_labels, instance_labels)
    loss = 0.5 * fg_loss + \
           0.5 * center_loss
    end_points['loss/fg_loss'] = fg_loss
    end_points['loss/center_loss'] = center_loss
    return loss,end_points
