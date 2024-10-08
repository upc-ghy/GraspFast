import torch.nn as nn
import torch
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from loss_utils import THRESH_GOOD, NUM_DIRECTION, NUM_ANGLE, NUM_DEPTH


scale_distribution_prior = np.load("ScaleDistribution/scale_distribution.npy", allow_pickle=True).item()
scale_distribution_prior_num = scale_distribution_prior['num']
intervals = scale_distribution_prior['interval'] 
scale_distribution_prior_ = torch.zeros(32)
for i in range(32):
    scale_distribution_prior_[i] = scale_distribution_prior_num[i]
scale_max_prob = torch.max(scale_distribution_prior_)
scale_distribution_prior_ = - (scale_distribution_prior_ / scale_max_prob).log() + 1
scale_distribution_prior = scale_distribution_prior_.cuda()

def generate_reweight_mask(end_points):
    top_direction_grasp_scores = end_points['batch_grasp_score']
    top_direction_grasp_widths = end_points['batch_grasp_width']
    B, Ns, A, D = top_direction_grasp_scores.size()
    top_direction_grasp_scores = top_direction_grasp_scores.reshape(B, Ns, -1)
    top_grasp_score_inds = torch.argmax(top_direction_grasp_scores, dim=2, keepdim=True)
    top_direction_grasp_widths = top_direction_grasp_widths.reshape(B, Ns, -1)
    top_widths = torch.gather(top_direction_grasp_widths, 2, top_grasp_score_inds).squeeze(2)
    
    id_mask = torch.zeros(size=top_widths.size()).long()
    for idx in range(len(intervals) - 1):
        id_mask[(intervals[idx] < top_widths) * (intervals[idx + 1] > top_widths)] = idx
    weight_mask = scale_distribution_prior[id_mask]  # (B, Ns)
    return weight_mask


def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    direction_loss, end_points = compute_direction_graspness_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    width_loss, end_points = compute_width_loss(end_points)
    loss = objectness_loss + 10 * graspness_loss + 100 * direction_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def get_weighted_loss(end_points):
    reweight_mask = generate_reweight_mask(end_points)
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    direction_loss, end_points = compute_weighted_direction_graspness_loss(end_points, reweight_mask.clone())
    score_loss, end_points = compute_weighted_score_loss(end_points, reweight_mask.clone())
    width_loss, end_points = compute_weighted_width_loss(end_points, reweight_mask.clone())
    loss = objectness_loss + 10 * graspness_loss + 100 * direction_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_direction_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    direction_score = end_points['direction_score']
    direction_label = end_points['batch_grasp_direction_graspness']
    loss = criterion(direction_score, direction_label)
    end_points['loss/stage2_direction_loss'] = loss
    return loss, end_points


def compute_weighted_direction_graspness_loss(end_points, weight_mask, width_weight_mask=None):
    criterion = nn.SmoothL1Loss(reduction='mean')
    direction_score = end_points['direction_score']
    direction_label = end_points['batch_grasp_direction_graspness']
    graspable_mask = end_points['graspable_mask']
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_direction)
    if not (width_weight_mask is None):
        weight_mask = (weight_mask * width_weight_mask)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_direction)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask = graspable_mask * weight_mask
    pos_direction_pred_mask = ((direction_score >= THRESH_GOOD) & graspable_mask)

    loss = criterion(direction_score, direction_label)
    loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

    end_points['loss/stage2_direction_loss'] = loss
    end_points['stage1_pos_direction_pred_count'] = pos_direction_pred_mask.long().sum() 

    return loss, end_points

# ？
def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_weighted_score_loss(end_points, weight_mask):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    graspable_mask = end_points['graspable_mask']
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask = graspable_mask * weight_mask
    loss = criterion(grasp_score_pred, grasp_score_label)

    loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points


def compute_weighted_width_loss(end_points, weight_mask):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    graspable_mask = end_points['graspable_mask']
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask1 = graspable_mask * weight_mask
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()

    loss = torch.sum(loss * loss_mask1) / (loss_mask1.sum() + 1e-6)
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
