#ifndef INCLUDE_OP_KERNELS_CPU_EMB_KERNEL_H_
#define INCLUDE_OP_KERNELS_CPU_EMB_KERNEL_H_

#include <base/base.h>
#include <tensor/tensor.h>

namespace kernel
{
    void emb_kernel_cpu
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        int32_t vocab_size,
        void* stream = nullptr
    );
}

#endif // INCLUDE_OP_KERNELS_CPU_EMB_KERNEL_H_