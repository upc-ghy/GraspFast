import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'torchKNN'))
from knn_modules import myknn

from loss_utils import GRASP_MAX_WIDTH, batch_directionpoint_params_to_matrix, transform_point_cloud, generate_grasp_directions


def process_grasp_labels(end_points):
    seed_xyzs = end_points['xyz_graspable']
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_directions_rot = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]
        poses = end_points['object_poses_list'][i]

        grasp_points_merged = []
        grasp_directions_rot_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]
            _, V, A, D = grasp_scores.size()
            num_grasp_points = grasp_points.size(0)
            grasp_directions = generate_grasp_directions(V).to(pose.device)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_directions_trans = transform_point_cloud(grasp_directions, pose[:3, :3], '3x3')
            angles = torch.zeros(grasp_directions.size(0), dtype=grasp_directions.dtype, device=grasp_directions.device)
            grasp_directions_rot = batch_directionpoint_params_to_matrix(-grasp_directions, angles)
            grasp_directions_rot_trans = torch.matmul(pose[:3, :3], grasp_directions_rot)

            grasp_directions_ = grasp_directions.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_directions_trans_ = grasp_directions_trans.transpose(0, 1).contiguous().unsqueeze(0)
            direction_inds = myknn(grasp_directions_trans_, grasp_directions_, k=1).squeeze() - 1
            grasp_directions_rot_trans = torch.index_select(grasp_directions_rot_trans, 0, direction_inds)
            grasp_directions_rot_trans = grasp_directions_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            grasp_scores = torch.index_select(grasp_scores, 1, direction_inds)
            grasp_widths = torch.index_select(grasp_widths, 1, direction_inds)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_directions_rot_merged.append(grasp_directions_rot_trans)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)
        grasp_directions_rot_merged = torch.cat(grasp_directions_rot_merged, dim=0)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)

        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        nn_inds = myknn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        grasp_directions_rot_merged = torch.index_select(grasp_directions_rot_merged, 0, nn_inds)
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)

        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_directions_rot.append(grasp_directions_rot_merged)
        batch_grasp_scores.append(grasp_scores_merged)
        batch_grasp_widths.append(grasp_widths_merged)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    batch_grasp_directions_rot = torch.stack(batch_grasp_directions_rot, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)

    direction_u_threshold = 0.6
    direction_grasp_num = 48
    batch_grasp_direction_valid_mask = (batch_grasp_scores <= direction_u_threshold) & (batch_grasp_scores > 0)
    batch_grasp_direction_valid = batch_grasp_direction_valid_mask.float()
    batch_grasp_direction_graspness = torch.sum(torch.sum(batch_grasp_direction_valid, dim=-1), dim=-1) / direction_grasp_num
    direction_graspness_min, _ = torch.min(batch_grasp_direction_graspness, dim=-1)
    direction_graspness_max, _ = torch.max(batch_grasp_direction_graspness, dim=-1)
    direction_graspness_max = direction_graspness_max.unsqueeze(-1).expand(-1, -1, 300)
    direction_graspness_min = direction_graspness_min.unsqueeze(-1).expand(-1, -1, 300)
    batch_grasp_direction_graspness = (batch_grasp_direction_graspness - direction_graspness_min) / (direction_graspness_max - direction_graspness_min + 1e-5)

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)
    batch_grasp_scores[~label_mask] = 0

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_direction_rot'] = batch_grasp_directions_rot
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_direction_graspness'] = batch_grasp_direction_graspness

    return end_points


def match_grasp_direction_and_label(end_points):
    top_direction_inds = end_points['grasp_top_direction_inds']
    template_directions_rot = end_points['batch_grasp_direction_rot']
    grasp_scores = end_points['batch_grasp_score']
    grasp_widths = end_points['batch_grasp_width']

    B, Ns, V, A, D = grasp_scores.size()
    top_direction_inds_ = top_direction_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_directions_rot = torch.gather(template_directions_rot, 2, top_direction_inds_).squeeze(2)
    top_direction_inds_ = top_direction_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_direction_grasp_scores = torch.gather(grasp_scores, 2, top_direction_inds_).squeeze(2)
    top_direction_grasp_widths = torch.gather(grasp_widths, 2, top_direction_inds_).squeeze(2)

    u_max = top_direction_grasp_scores.max()
    po_mask = top_direction_grasp_scores > 0
    po_mask_num = torch.sum(po_mask)
    if po_mask_num > 0:
        u_min = top_direction_grasp_scores[po_mask].min()
        top_direction_grasp_scores[po_mask] = torch.log(u_max / top_direction_grasp_scores[po_mask]) / (torch.log(u_max / u_min) + 1e-6)

    end_points['batch_grasp_score'] = top_direction_grasp_scores
    end_points['batch_grasp_width'] = top_direction_grasp_widths

    return top_template_directions_rot, end_points
