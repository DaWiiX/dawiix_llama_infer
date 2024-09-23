#ifndef INCLUDE_OP_KERNELS_INTERFACE_H_
#define INCLUDE_OP_KERNELS_INTERFACE_H_
#include <base/cuda_config.h>
#include "tensor/tensor.h"

namespace kernel
{
    using AddKernel = void (*)
    (
        const tensor::Tensor &input1,
        const tensor::Tensor &input2,
        const tensor::Tensor &output,
        void *stream
    );

    using MatmulKernel = void (*)
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        float scale,
        const CudaConfig* config
    );

    using MatmulKernelQuant = void (*)
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        int32_t group_size,
        const tensor::Tensor& scale, 
        const CudaConfig* config
    );

    using EmbeddingKernel = void (*)
    (
        const tensor::Tensor& input, 
        const tensor::Tensor& weight,
        const tensor::Tensor& output, 
        int32_t vocab_size, 
        void* stream
    );

    using SwigluKernel = void (*)
    (
        const tensor::Tensor& input1, 
        const tensor::Tensor& input2,
        const tensor::Tensor& output, 
        void* stream
    );

    using RMSNormKernel = void (*)
    (
        const tensor::Tensor &input,
        const tensor::Tensor &weight,
        const tensor::Tensor &output,
        void *stream
    );

    AddKernel get_add_kernel(base::DeviceType device_type);

    MatmulKernel get_matmul_kernel(base::DeviceType device_type);

    MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

    EmbeddingKernel get_emb_kernel(base::DeviceType device_type);
    
    SwigluKernel get_swiglu_kernel(base::DeviceType device_type);

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

}  // namespace kernel

#endif // INCLUDE_OP_KERNELS_INTERFACE_H_