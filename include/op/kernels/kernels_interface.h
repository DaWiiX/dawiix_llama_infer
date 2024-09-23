#ifndef INCLUDE_OP_KERNELS_INTERFACE_H_
#define INCLUDE_OP_KERNELS_INTERFACE_H_
#include <base/cuda_config.h>
#include "tensor/tensor.h"

namespace kernel
{
    using RMSNormKernel = void (*)
    (
        const tensor::Tensor &input,
        const tensor::Tensor &weight,
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

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

    MatmulKernel get_matmul_kernel(base::DeviceType device_type);

    MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

    
}

#endif // INCLUDE_OP_KERNELS_INTERFACE_H_