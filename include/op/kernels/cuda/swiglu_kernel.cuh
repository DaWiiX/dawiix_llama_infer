#ifndef INCLUDE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH_
#define INCLUDE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH_
#include <tensor/tensor.h>

namespace kernel
{
    void swiglu_kernel_cu
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output,
        void* stream
    );
}  // namespace kernel

#endif  // INCLUDE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH_