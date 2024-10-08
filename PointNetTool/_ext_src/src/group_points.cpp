// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"
#include "utils.h"

// 该函数的作用就是将原始点云points中的数据，依照采样出的npoints=1024个中心点邻域内筛选出的nsample=64个点的索引位置idx，
// 在原始点云points中提取原始数据保存到out张量中。
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

// group_points() 函数会根据 idx 中的下标值，从 points 中选出相应的坐标值，
// 并按照指定的顺序组合成一个形状为 [B, C, M, K] 的 FloatTensor，作为函数的返回值。
at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);  // 确保原始点云数据是连续存储在内存空间中的，(B, 3, N)
  CHECK_CONTIGUOUS(idx);     // idx维度是(B, npoint=1024, nsample=64)
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  // 这个是一个维度为(B, 3, npoint=1024, nsample=64)维度的 0 值Tensor张量
  // output是一个结果张量，用于存储分组好的点云的下标
  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    // points.size(0)=B是batch size, points.size(1)=3是xyz三通道数, points.size(2)=N是原始点云的数目
    // idx.size(1)=npoint=1024个中心点也就是抓取点，idx.size(2)=64是每个采样点邻域内筛选出的64个点
    // 该函数的作用就是将原始点云points中的数据，依照采样出的npoints=1024个中心点邻域内筛选出的nsample=64个点的索引位置idx，在原始点云points中提取原始数据保存到out张量中。
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;  // (B, C, npoint, nsample)
}


at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}
