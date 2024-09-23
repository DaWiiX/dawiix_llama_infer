#ifndef INCLUDE_OP_KERNELS_CPU_SWIGLU_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_SWIGLU_KERNEL_H_
#include <tensor/tensor.h>

namespace kernel
{
    void swiglu_kernel_cpu
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output,
        void* stream
    );
}  // namespace kernel

#endif  // INCLUDE_OP_KERNELS_CPU_SWIGLU_KERNEL_H_