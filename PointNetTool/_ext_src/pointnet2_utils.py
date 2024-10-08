# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):  # features表示原始的点云数据维度为[B, 3, N],idx表示中心中心点需要采样的点下标,维度为[B,npoint=1024,sample=64]
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()  # B是batch size大小，nfeatures是1024中心点数目，nsample表示每个中心点采样64个点数目
        _, C, N = features.size()           # C是x,y,z三维通道数目3，N是点的数目三千多个原始点 
        
        # 将idx和N组成元组存放到上下文对象ctx的for_backwards属性中，便于后面backward()方法使用上下文对象中的属性值
        ctx.for_backwards = (idx, N)

        # 该函数的作用就是将原始点云features中的数据，依照采样出的npoints=1024个中心点邻域内筛选出的nsample=64个点的索引位置idx，
        # 在原始点云features中提取原始数据保存到一个张量中返回，这return的就是一个维度为(B, C, npoint, nsample)的张量
        # 其中B是batch size, C是xyz三通道，npoint是中心点数目1024， nsample=64是每个中心点邻域内筛选的点数64
        return _ext.group_points(features, idx)  # (B, C, npoint, nsample)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class CylinderQuery(Function):

    # radius=0.05, hmin=-0.02, hmax=[0.01, 0.02, 0.03, 0.04], nsample=64, xyz是输入的点云数据, new_xyz是中心点的点云数据, rot把中心点的旋转矩阵3*3的拉平
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the cylinders
        hmin, hmax : float
            hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]
            endpoints of cylinder height in x-rotation axis
        nsample : int
            maximum number of features in the cylinders
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the cylinder query
        rot: torch.Tensor
            (B, npoint, 9) flatten rotation matrices from
                           cylinder frame to world frame  # 将旋转矩阵从圆柱体坐标系中flatten到世界坐标系中

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        # hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]
        # xyz是输入的点云数据, new_xyz是中心点的点云数据, 当前rot把中心点3*3的旋转矩阵拉平的矩阵（该旋转矩阵是将中心点的圆柱体坐标系转换成世界坐标系）
        return _ext.cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None

cylinder_query = CylinderQuery.apply


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height
    该类的作用是将一个点云分组到以某个圆柱体为查询对象的区域内。
    其中radius表示圆柱体的半径，hmin和hmax表示圆柱体高度的两端点，nsample下一层AbstractSet层的该分组的点云数目。
    forward方法接受原始的点集xyz、查询圆柱体的中心点new_xyz、对应的旋转矩阵rot和特征集features，
    并输出新的特征集new_features，其中查询圆柱体和分组区域通过idx指示。
    如果use_xyz为True，则将坐标也作为特征一并考虑；如果ret_grouped_xyz为True，
    则返回grouped_xyz表示每个点对应的相对坐标；如果normalize_xyz为True，
    则将grouped_xyz归一化到半径上；如果rotate_xyz为True，则在输出new_features之前进行旋转变换。

    Parameters
    ---------
    radius : float32
        radius=0.05
        Radius of cylinder
    hmin, hmax: float32
        hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04]
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        nsample=64
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        # type: (CylinderQueryAndGroup, float, float, float, int, bool) -> None
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz                    #use_xyz=True
        self.ret_grouped_xyz = ret_grouped_xyz    # re_grouped_xyz=False
        self.normalize_xyz = normalize_xyz        # normalize_xyz=False
        self.rotate_xyz = rotate_xyz              # rotate_xyz=True
        self.sample_uniformly = sample_uniformly  # sample_uniformly=False
        self.ret_unique_cnt = ret_unique_cnt      # ret_unique_cnt=False
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    # 传过来的参数中，xyz是原始的点云数据，其数目远超1024, 维度为[B, N, 3]
    # new_xyz就是采样的1024个点组成的点云，维度为[B, 1024, 3]
    # rot就是这个1024个点对应的旋转矩阵[B, 1024, 3, 3]
    def forward(self, xyz, new_xyz, rot, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3), 其中N是远超三千多个原始点
        new_xyz : torch.Tensor
            centriods (B, npoint, 3), npoint是点云
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3), npoint表示采样的1024个中心点
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        # radius=0.05, hmin=-0.02, hmax=[0.01, 0.02, 0.03, 0.04], nsample=64, xyz是输入的点云数据, new_xyz是中心点的点云数据,
        # cylinder_query就是查询以self.radius为半径，以hmin为爪子原本深度，以self.nsample表示每个中心点要采样的邻域范围内的点的数目，
        # 以new_xyz表示[B, 1024, 3]的中心点，xyz表示原始点云数据[B, N, 3]，从原始点云数据中筛选出中心点邻域的点
        # rot.veiw()把中心点的旋转矩阵3*3的拉平
        # 返回的是当前depth下中心点邻域范围内采样的64个点的索引值，维度为[B, npoint=1024, self.nsample=64]
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:  # sample_uniformly=False
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        # xyz原始的点云数据维度是[B, N, 3]，转换后的维度是[B, 3, N], 将张量存放到连续的地址中
        xyz_trans = xyz.transpose(1, 2).contiguous()
        
        # 输入维度转换成[B, 3, N]的点云数据xyz_trans，以及当前depth下1024个中心点每个采样64个点的下标idx，维度为[B, npoint=1024, nsample=64]
        # 输出的是：将原始点云数据xyz_trans中的数据，依照采样出的npoints=1024个中心点邻域内筛选出的nsample=64个点的索引位置idx，
        # 在原始点云xyz_trans中提取索引位置的原始数据保存到张量grouped_xyz中
        grouped_xyz = grouping_operation(xyz_trans, idx)      # (B, 3, npoint, nsample)
        # new_xyz表示维度为(B, 1024, 3)的中心点， 通过.transpose(1, 2)转换成维度为(B, 3, 1024), 
        # 通过.unsqueeze(-1)在最后一维度增加一维变成(B, 3, 1024, 1)
        # 维度为(B, 3, 1024, 64)的中心点邻域点数据grouped_xyz 加上 中心点坐标变换后的(B, 3, 1024, 1)张量，首先，
        # 中心点坐标的张量(B, 3, 1024, 1)会把(B, 3, 1024)三个维度的所有数据复制64份，也就是64份相同的张量(B, 3, 1024)拼成一个新张量(B, 3, 1024, 64)
        # 然后相同维度的相加，比如grouped_xyz[1][2][1000][9]与新张量[1][2][1000][9]位置相加
        # 整套操作就是把中心点的xyz坐标值加到它对应的邻域点的xyz坐标值上，例如：中心点x+某一个邻域点x
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:  # self.normalize_xyz = False
            grouped_xyz /= self.radius
        if self.rotate_xyz:     # self.rotate_xyz = True
            # 首先将整合后附加了中心点坐标的邻域点坐标张量转换维度：(B, 3, 1024, 64) -> (B, 1024, 64, 3)
            # 然后通过.contiguous()将张量放到连续的内存中间中
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous()  # (B, npoint, nsample, 3)
            # rot就是这个1024个点对应的旋转矩阵[B, 1024, 3, 3]
            # 维度为(B, 1024, 64, 3)的矩阵乘以(B, 1024, 3, 3)也就是(B, 1024, 64, 3)
            # 也就是对整合了中心点坐标的邻域点坐标进行旋转到。。坐标系
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)               # torch.matmul()表示矩阵乘法
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()  # 又把grouped_xyz_从维度为(B, 1024, 64, 3)转换为(B, 3, 1024, 64)

        if features is not None:  # 第一次训练好像是空的，没有传参过来，默认是None
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz  # self.use_xyz = True
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz  # 使用获得的整合了中心点坐标的邻域点坐标作为当前最新的特征张量new_features

        ret = [new_features]      # 一个张量作为列表中的一个元素，ret列表中初始只有一个张量new_features，维度为(B, 3, 1024, 64)
        if self.ret_grouped_xyz:  #  self.ret_grouped_xyz = False
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:   # ret_unique_cnt=False
            ret.append(unique_cnt)
        if len(ret) == 1:         # 列表中只有一个张量new_features，故满足len(ret)==1
            return ret[0]         # ret[0]就表示张量new_features,返回的是一个张量
        else:
            return tuple(ret)