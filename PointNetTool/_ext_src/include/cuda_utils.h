// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 512

// work_size = npoint（中心点的数目）
inline int opt_n_threads(int work_size) {
  // 求出work_size是2的几次方，也就是log_{2}{work_size}
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  // 1<<pow_2 = 2^{pow_2}, 每位移一次相当于乘以2，这个是位移了pow_2次
  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
  const int x_threads = opt_n_threads(x);
  const int y_threads =
      max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
  dim3 block_config(x_threads, y_threads, 1);

  return block_config;
}

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                   \
  } while (0)

#endif
