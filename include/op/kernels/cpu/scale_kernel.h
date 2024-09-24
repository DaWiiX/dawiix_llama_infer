#ifndef INCLUDE_OP_KERNELS_CPU_SCALE_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_SCALE_KERNEL_H_

#include <tensor/tensor.h>

namespace kernel
{
    void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}  // namespace kernel

#endif  // INCLUDE_OP_KERNELS_CPU_SCALE_KERNEL_H_