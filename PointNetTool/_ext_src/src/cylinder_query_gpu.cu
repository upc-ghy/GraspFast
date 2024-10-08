// Author: chenxi-wang

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// 创建一个核函数query_cylinder_point_kernel
// hmin=-0.02
// hmax=[0.01, 0.02, 0.03, 0.04]
// radius = 0.05
// b=B（Batch size）,n=N(点云得数目),m=npoint（中心点的数目）
// hmin=-0.02
// hmax=[0.01, 0.02, 0.03, 0.04]
// radius = 0.05
// nsample=64
// xyz是输入的点云数据, new_xyz是中心点的点云数据, 当前rot把中心点3*3的旋转矩阵拉平的矩阵（该旋转矩阵是将中心点的圆柱体坐标系转换成世界坐标系）
// idx就是创建了一个（B, npoint, 64）大小的 0 tensor矩阵
__global__ void query_cylinder_point_kernel(int b, int n, int m, float radius, float hmin, float hmax,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        const float *__restrict__ rot,
                                        int *__restrict__ idx) {
  // 一个batch size分配了一个线程块，因此blockIdx.x的下标值就是第几块的batch的下标值
  int batch_index = blockIdx.x;
  // 一个batch的点云中点的数目就是n个，每个点3维度（x,y,z），故每个batch指向该批点云地址的指针地址就是，batch的索引*一个batch的点数目*每个点的维度
  xyz += batch_index * n * 3;
  // 每个线程块中，中心点云数据的起始指针同上
  new_xyz += batch_index * m * 3;
  // 旋转矩阵是3*3的，所以乘了9，同上
  rot += batch_index * m * 9;
  // 每一批中，每个中心点采样都采样nsample个点，m个中心点采样m*nsample
  idx += batch_index * m * nsample;
  
  int index = threadIdx.x;   // 当前线程的id号
  int stride = blockDim.x;   // 线程块在x轴方向上的大小，也就是在x轴方向的线程块中包含多少个线程

  float radius2 = radius * radius;  // 半径0.05的平方

  // 起始j是当前线程的id，j小于中心点的个数，j每次增加一个块中的线程数量，也就是j一次跳一块线程块
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];  // 利用中心点点云的指针对象new_xyz获取下标为j的中心点的x, y, z三个坐标的数据
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float r0 = rot[j * 9 + 0];  // 中心点在夹爪坐标系向世界坐标系转换的矩阵，所以乘以9
    float r1 = rot[j * 9 + 1];
    float r2 = rot[j * 9 + 2];
    float r3 = rot[j * 9 + 3];
    float r4 = rot[j * 9 + 4];
    float r5 = rot[j * 9 + 5];
    float r6 = rot[j * 9 + 6];
    float r7 = rot[j * 9 + 7];
    float r8 = rot[j * 9 + 8];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      // 坐标在各个方向上的差值，点云坐标是基于什么坐标系的？这两相当于两个向量的差值吧
      float x = xyz[k * 3 + 0] - new_x;
      float y = xyz[k * 3 + 1] - new_y;
      float z = xyz[k * 3 + 2] - new_z;
      // 这里是一个差值向量右乘一个旋转矩阵，右乘是绕着自身各轴的旋转。
      float x_rot = r0 * x + r3 * y + r6 * z;   // 这个求出来的是approaching distance吗?是
      float y_rot = r1 * x + r4 * y + r7 * z;
      float z_rot = r2 * x + r5 * y + r8 * z;
      float d2 = y_rot * y_rot + z_rot * z_rot;
      if (d2 < radius2 && x_rot > hmin && x_rot < hmax) { // 这里是
        if (cnt == 0) {  // 只要还没采样到一个满足范围内的点，只要采样到一个满足条件的，立马把这个中心点j需要采样的nsample点都设置为当前满足范围的点k
          for (int l = 0; l < nsample; ++l) {  // 将中心点j采样的这一个nsample数量的点都设置为当前满足范围内的点k（k是点的下标）
            idx[j * nsample + l] = k;
          }
        }
        // 后面如果出现其它满足的，然后再挨个改
        idx[j * nsample + cnt] = k; // j是中心点的坐标，nsample是在中线点要采样的点数
        ++cnt;
      }
    }
  }
}


// b=B（Batch size）,n=N(点云得数目),m=npoint（中心点的数目）
// hmin=-0.02
// hmax=[0.01, 0.02, 0.03, 0.04]
// radius = 0.05
// nsample=64
// xyz是输入的点云数据, new_xyz是中心点的点云数据, 当前rot把中心点3*3的旋转矩阵拉平的矩阵（该旋转矩阵是将中心点的圆柱体坐标系转换成世界坐标系）
// idx就是创建了一个（B, npoint, 64）大小的 0 tensor矩阵
// 其中参数float *xyz指针接收的是指向点云数据第一个点地址的指针
void query_cylinder_point_kernel_wrapper(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // b表示线程块的数目
  // 其中opt_n_threads(m)是获取比m小的离m最近的2的指数，表示线程的数目
  // 0 表示共享内存大小，表示当前 kernel 不需要使用共享内存；
  // stream 表示当前 kernel 所在的 CUDA 流。
  query_cylinder_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, hmin, hmax, nsample, new_xyz, xyz, rot, idx);
  
  CUDA_CHECK_ERRORS();
}
