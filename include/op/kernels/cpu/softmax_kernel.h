#ifndef INCLUDE_OP_KERNELS_CPU_SOFTMAX_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_SOFTMAX_KERNEL_H_

#include "tensor/tensor.h"

namespace kernel
{
    void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
} // namespace kernel

#endif // INCLUDE_OP_KERNELS_CPU_SOFTMAX_KERNEL_H_