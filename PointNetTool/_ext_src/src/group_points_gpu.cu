// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// 这个核函数的作用就是把原始点云中的数据筛选出的下标值，存放到out中
// input: points(b, c=3, n) idx(b, npoints=1024, nsample)
// output: out(b, c, npoints, nsample)
// __restrict__ 是 C 语言的一个关键字，用于提示编译器变量指针是唯一能够访问到特定内存区域的指针，
// __restrict__ 修饰符告诉编译器，该指针所对应的内存区域不会被其他指针访问或修改，
// 即没有别的指针能够访问或修改 points 所指向的数组中的元素，从而使编译器可以基于这种假设做一些优化来提高程序的性能
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;   // 当前是哪一个block块，返回它的id，这个id就是batch的当前索引值，比如b个batch，现在是处理第batch_index+1个batch
  
  // 原始点数据，筛选点下标位置，筛选点要存放在结果张量out中的位置
  points += batch_index * n * c;  // 原始点点云数据规模是b*c*n, 现在处理的是第batch_index+1个batch, 所以指针移动到当前batch, 一个batch的数据数目是n*c
  idx += batch_index * npoints * nsample;     // 筛选的点的下标
  out += batch_index * npoints * nsample * c; // 该点在结果张量out中的位置 

  // Block是b*1*1的，每个Block中的线程是x*y*1的
  // 这算的是在某一个block上的当前线程索引值
  const int index = threadIdx.y * blockDim.x + threadIdx.x;  // 记住线程各个维度的情况，因此是乘y维度的再加上x维度的
  const int stride = blockDim.y * blockDim.x;                // 一个block上线程的总数目，也就是后面一次跳一个block
  // i表示当前的线程索引值, i每次跳一个block, i要小于3*1024
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;   // 除法，取下整; 每个batch中的npoints=1024个点对应这当前batch中的n个原始点；这 l 算出来有点像batch的索引值的感觉
    const int j = i % npoints;   // 取余数，也就是算出了当前是哪一个中心点的索引值
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];    // ii是当前中心点对应的筛选出的邻域点的索引，前j个中心点每个nsample个邻域点，当前到了第k个，总索引值就是先乘再加
      // 其中out的维度是[B, 3, npoints, nsample]
      // 依据上面一行代码算出的筛选点在原始点云[B, 3, n]中的具体位置l*n+ii, 并将取出的结果赋值给
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

// b表示batch size, c表示xyz三通道数，n表示原始点云中点的数目三千多
// npoints表示采样的中心点的数目1024, nsample就是每个中心点筛选出的点数目64
// *points就是原始点云数目，维度为 (b, c=3, n)
// *idx的表示要筛选出的点的下标 (b, npoints=1024, nsample=64)
// *out就是保存结果的张量，维度为 (b, c=3, npoints=1024, nsample=64)
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out) {
  // 这段代码使用了 PyTorch 中的 at::cuda::getCurrentCUDAStream() 函数，该函数返回当前线程所在的 CUDA 流（stream）句柄
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 调用核函数, 申请的大小为b*1*1规格的Block，x_threads*y_threads*1规格的Thread;
  // 表示共享内存大小，即为每个线程块分配的共享内存大小。共享内存是指在同一个线程块内的线程之间共享的内存空间，可以用来实现线程之间的协作和通信
  // stream表示核函数执行所在的 CUDA 流句柄。CUDA 流是一组由操作排队的序列，可以使多个 GPU 操作同时进行，从而提高并行计算效率
  // 该核函数的作用就是将原始点云points中的数据，依照采样出的npoints=1024个中心点邻域内筛选出的nsample=64个点的索引位置idx，在原始点云points中提取原始数据保存到out张量中。
  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}
