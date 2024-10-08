// Author: chenxi-wang

#include "cylinder_query.h"
#include "utils.h"

void query_cylinder_point_kernel_wrapper(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx);
// xyz是输入的点云数据, new_xyz是中心点的点云数据, 当前rot把中心点3*3的旋转矩阵拉平的矩阵（该旋转矩阵是将中心点的圆柱体坐标系转换成世界坐标系）
at::Tensor cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, const float radius, const float hmin, const float hmax,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);  // 在深度学习模型开发中，由于计算机的内存管理机制可能导致非连续的张量造成性能上的损失，因此需要确保所有的张量都是连续的，以避免这种情况的发生。
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(rot);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(rot);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(rot);
  }

// nex_xyz.size(0)=B（batch size）, new_xyz.size(1)=npoint (中心点的数目), nsample=64（每个圈内采样的样本个数）
// idx就是创建了一个（B, npoint, 64）大小的 0 tensor矩阵
  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    //xyz.size(0)=B（Batch size）,xyz.size(1)=N(原始点云的数目),new_xyz.size(1)=npoint（中心点的数目1024）
    // hmin=-0.02
    // hmax=[0.01, 0.02, 0.03, 0.04]
    // radius = 0.05
    // nsample=64
    // xyz是输入的点云数据, new_xyz是中心点的点云数据, 当前rot把中心点3*3的旋转矩阵拉平的矩阵（该旋转矩阵是将中心点的圆柱体坐标系转换成世界坐标系）
    // idx就是创建了一个（B, npoint, 64）大小的 0 tensor矩阵
    // xyz.data<float>()获得的是指向xyy点云tensor张量的第一个点的地址的指针，该指针是一个float* 类型（点云中数据是float类型的）
    query_cylinder_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, hmin, hmax, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), rot.data<float>(), idx.data<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return idx;  // cylinder_query(), 返回的是当前depth下，中心点所有的圆柱体范围内的64个点索引值，维度大小为(B, npoint, 64), 其中npoint表示中心点的数目1024个
}
