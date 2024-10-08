import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCE(nn.Module):
    def __init__(self, feat_channel):
        super().__init__()
        self.in_dim = feat_channel
        self.conv_graspable = nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)
        end_points['graspable_score'] = graspable_score
        return end_points



class PCS(nn.Module):
    def __init__(self, feat_channel):
        super().__init__()
        self.in_dim = feat_channel
        self.conv_object = nn.Conv1d(self.in_dim, 2, 1)

    def forward(self, seed_features, end_points):
        object_criteria = self.conv_object(seed_features)
        end_points['object_criteria'] = object_criteria
        return end_points