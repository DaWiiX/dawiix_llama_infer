#ifndef INCLUDE_OP_KERNELS_CPU_SCALE_SUM_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_SCALE_SUM_KERNEL_H_

#include <tensor/tensor.h>

namespace kernel
{
    void scale_sum_kernel_cpu
    (
        const tensor::Tensor& value, 
        const tensor::Tensor& scale, 
        const tensor::Tensor& output, 
        int t, 
        int d, 
        int stride,
        void* stream = nullptr
    );
} // namespace kernel
#endif  // INCLUDE_OP_KERNELS_CPU_SCALE_SUM_KERNEL_H_