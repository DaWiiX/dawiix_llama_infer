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

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
}

#endif // INCLUDE_OP_KERNELS_INTERFACE_H_