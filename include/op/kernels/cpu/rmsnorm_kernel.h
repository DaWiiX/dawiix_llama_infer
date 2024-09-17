#ifndef INCLUDE_OP_KERNELS_CPU_RMSNORM_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_RMSNORM_KERNEL_H_
#include "tensor/tensor.h"

namespace kernel
{
    void rmsnorm_kernel_cpu(
        const tensor::Tensor &input,
        const tensor::Tensor &weight,
        const tensor::Tensor &output,
        void *stream = nullptr);
} // namespace kernel

#endif // INCLUDE_OP_KERNELS_CPU_RMSNORM_KERNEL_H_