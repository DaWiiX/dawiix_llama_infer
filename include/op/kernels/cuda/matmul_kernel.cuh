#ifndef INCLUDE_OP_KERNELS_CU_MATMUL_KERNEL_H_
#define INCLUDE_OP_KERNELS_CU_MATMUL_KERNEL_H_
#include "tensor/tensor.h"

namespace kernel
{
    void matmul_kernel_cu
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        float scale = 1.f,
        const CudaConfig* config = nullptr
    );

    void matmul_kernel_cu_qint8
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        int32_t group_size,
        const tensor::Tensor& scale, 
        const CudaConfig* config = nullptr
    );
} // namespace kernel

#endif // INCLUDE_OP_KERNELS_CPU_MATMUL_KERNEL_H_