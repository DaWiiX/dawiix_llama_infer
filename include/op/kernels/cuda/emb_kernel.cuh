#ifndef INCLUDE_OP_KERNELS_CUDA_EMB_KERNEL_CUH_
#define INCLUDE_OP_KERNELS_CUDA_EMB_KERNEL_CUH_

#include "tensor/tensor.h"

namespace kernel 
{
    void emb_kernel_cu
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        int32_t vocab_size, 
        void* stream = nullptr
    );
}
#endif // INCLUDE_OP_KERNELS_CUDA_EMB_KERNEL_CUH_