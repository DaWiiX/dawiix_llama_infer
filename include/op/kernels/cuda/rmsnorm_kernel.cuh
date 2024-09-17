#ifndef INCLUDE_OP_KERNELS_CUDA_RMSNORM_KERNEL_CUH_
#define INCLUDE_OP_KERNELS_CUDA_RMSNORM_KERNEL_CUH_

#include <tensor/tensor.h>

namespace kernel
{
    void rmsnorm_kernel_cu
    (
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        void* stream = nullptr
    );
}

#endif  // INCLUDE_OP_KERNELS_CUDA_RMSNORM_KERNEL_CUH_