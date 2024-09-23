#ifndef INCLUDE_OP_KERNELS_CUDA_ADD_CU_H_
#define INCLUDE_OP_KERNELS_CUDA_ADD_CU_H_
#include "tensor/tensor.h"

namespace kernel
{
    void add_kernel_cu
    (
        const tensor::Tensor &input1, 
        const tensor::Tensor &input2,
        const tensor::Tensor &output, 
        void *stream = nullptr
    );
} // namespace kernel
#endif // INCLUDE_OP_KERNELS_CUDA_ADD_CU_H_
