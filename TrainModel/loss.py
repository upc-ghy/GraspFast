import torch.nn as nn
import torch
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from loss_utils import THRESH_GOOD, NUM_VIEW, NUM_ANGLE, NUM_DEPTH

# scale distribution prior
scale_distribution_prior = np.load("statistic/scale_distribution.npy", allow_pickle=True).item()
scale_distribution_prior_num = scale_distribution_prior['num']   # 按照目标物的尺寸进行了分组，尺寸从小到大，每组多少个采样可抓取点数，32组
intervals = scale_distribution_prior['interval']    # 尺度的间隙坐标，33个
scale_distribution_prior_ = torch.zeros(32)
for i in range(32):
    scale_distribution_prior_[i] = scale_distribution_prior_num[i]    # 取出scale_distribution_prior_num中的每一个数目，再存一遍
scale_max_prob = torch.max(scale_distribution_prior_)     # 取出哪个尺度下可抓取点数最多的那个数目scale_max_prob
scale_distribution_prior_ = - (scale_distribution_prior_ / scale_max_prob).log() + 1   # 这个就是论文中那个多尺度权重公式了。。。。。
scale_distribution_prior = scale_distribution_prior_.cuda()


# 依据每个点V*A*D种位姿中的置信分数最高的位姿对应的抓取宽度，利用该点该最优抓取宽度对应的目标物尺度区间，获取该点的多尺度权重
def generate_reweight_mask(end_points):
    top_view_grasp_scores = end_points['batch_grasp_score']  # (B, Ns, A, D)
    top_view_grasp_widths = end_points['batch_grasp_width']  # (B, Ns, A, D)
    B, Ns, A, D = top_view_grasp_scores.size()   # 获取每个点对应的最优抓取方向下的旋转角度和旋转维度对应的抓取位姿的grasp score的维度
    top_view_grasp_scores = top_view_grasp_scores.reshape(B, Ns, -1)  # (B, Ns, A*D): reshape之后好求每个点最优的抓取位姿的索引
    top_grasp_score_inds = torch.argmax(top_view_grasp_scores, dim=2, keepdim=True)  # (B, Ns, 1) 取出每一个点置信分数最高的那个抓取位姿的索引
    top_view_grasp_widths = top_view_grasp_widths.reshape(B, Ns, -1)  # (B, Ns, A, D) -> (B, Ns, A*D)
    top_widths = torch.gather(top_view_grasp_widths, 2, top_grasp_score_inds).squeeze(2)  # (B, Ns) 得到每个点最优抓取位姿对应的抓取宽度
    
    id_mask = torch.zeros(size=top_widths.size()).long()  # 创建一个(B, Ns)维度的全是0的张量
    # 由每个点的最优抓取宽度，确定落在的目标物尺度范围。
    for idx in range(len(intervals) - 1):  # intervals表示目标物尺度间隙，len(intervals)有33个间隙，这里减去1表示32个种尺度
        # 若(B, Ns)维度的抓取宽度某个位置的宽度在当前目标物尺度范围内，则id_mask相同位置由0改为当前尺度索引
        id_mask[(intervals[idx] < top_widths) * (intervals[idx + 1] > top_widths)] = idx
    # 对于每个元素id_mask[i, j], 将会选择scale_distribution_prior的32个权重中第id_mask[i,j]个元素作为weight_mask中的第(i, j)个元素
    weight_mask = scale_distribution_prior[id_mask]  # (B, Ns)
    return weight_mask


def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    view_loss, end_points = compute_view_graspness_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    width_loss, end_points = compute_width_loss(end_points)
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def get_weighted_loss(end_points):
    reweight_mask = generate_reweight_mask(end_points)
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    view_loss, end_points = compute_weighted_view_graspness_loss(end_points, reweight_mask.clone())
    score_loss, end_points = compute_weighted_score_loss(end_points, reweight_mask.clone())
    width_loss, end_points = compute_weighted_width_loss(end_points, reweight_mask.clone())
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    # 几个批次的样本损失进行平均，使得交叉熵损失对样本数量不敏感
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']  # 这是GraspNet预测出的关于objectness的两个指标(B, 2, point_num)
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)  # 在维度为1的轴上找到一个物体两个objectness_score中最大值所在的索引，索引只有两个0和1, 1位置上的表示可抓取， 0位置上的表示不可抓取
    # 如果预测的目标物是否可抓取和真实的是否可抓取标签(0或者1)一致，则表示预测精确，输出精确率。 .long()是转换为64位整数型  .mean()表示该批次所有点的抑制率的平均值
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    # 精确率，预测为1 (objectness_pred == 1)中，确实标签也是1的所占的比率
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    # 召回率，标签为1中，预测的也1的概率
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
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


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


# 增加多尺度权重后的损失抓取视角的损失计算
def compute_weighted_view_graspness_loss(end_points, weight_mask, width_weight_mask=None):
    criterion = nn.SmoothL1Loss(reduction='mean')
    # 获取预测的抓取视角得分
    view_score = end_points['view_score']   # (B, Ns, V)
    # 对应的真实标签
    view_label = end_points['batch_grasp_view_graspness']  # (B, Ns, V)
    # 因为计算多尺度权重的前提是点是可抓取点，因此取出保存的点的可抓取掩码
    graspable_mask = end_points['graspable_mask']  # (B, point_num)
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_VIEW) # (B, point_num, NUM_VIEW)
    if not (width_weight_mask is None):
        weight_mask = (weight_mask * width_weight_mask)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_VIEW)  #每个点的多尺度权重(B, point_num)->(B, point_num, 1)->(B, point_num, NUM_VIEW)
    # graspable_mask = graspable_mask.to(weight_mask.device)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask = graspable_mask * weight_mask  # (B, point_num, NUM_VIEW), 每个点是否可抓取，对可抓取的点施加该点的多尺度权重
    pos_view_pred_mask = ((view_score >= THRESH_GOOD) & graspable_mask)

    loss = criterion(view_score, view_label)

    # 对计算出的损失中可抓取点的损失施加多尺度权重
    loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

    end_points['loss/stage2_view_loss'] = loss
    # 保存，对于可抓取点其抓取方向得分超过THRESH_GOOD=0.7的抓取方向的数目
    end_points['stage1_pos_view_pred_count'] = pos_view_pred_mask.long().sum()  # (B, Ns, V)

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
    grasp_score_pred = end_points['grasp_score_pred']    # (B, Ns, A, D)
    grasp_score_label = end_points['batch_grasp_score']  # (B, Ns, A, D)
    # 因为计算多尺度权重的前提是点是可抓取点，因此取出保存的点的可抓取掩码
    graspable_mask = end_points['graspable_mask']  # (B, point_num)
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)  # (B, point_num, A, D)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)  # (B, point_num, A, D)
    # graspable_mask = graspable_mask.to(weight_mask.device)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask = graspable_mask * weight_mask    # (B, point_num, A, D)
    loss = criterion(grasp_score_pred, grasp_score_label)

    # 对计算出的损失中可抓取点的损失施加多尺度权重
    loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']    # (B, Ns, A, D)
    grasp_width_label = end_points['batch_grasp_width'] * 10   # (B, Ns, A, D)
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points


def compute_weighted_width_loss(end_points, weight_mask):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']    # (B, Ns, A, D)
    grasp_width_label = end_points['batch_grasp_width'] * 10   # (B, Ns, A, D)
    # 因为计算多尺度权重的前提是点是可抓取点，因此取出保存的点的可抓取掩码
    graspable_mask = end_points['graspable_mask']  # (B, point_num)
    graspable_mask = graspable_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)  # (B, point_num, A, D)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, NUM_ANGLE, NUM_DEPTH)  # (B, point_num, A, D)
    # graspable_mask = graspable_mask.to(weight_mask.device)
    weight_mask = weight_mask.to(graspable_mask.device)
    loss_mask1 = graspable_mask * weight_mask    # (B, point_num, A, D)
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()

    # 对计算出的损失中可抓取点的损失施加多尺度权重
    loss = torch.sum(loss * loss_mask1) / (loss_mask1.sum() + 1e-6)
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
