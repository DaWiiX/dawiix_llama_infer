#include "base/base.h"
#include "op/kernels/kernels_interface.h"

#include "op/kernels/cpu/rmsnorm_kernel.h"
#include "op/kernels/cuda/rmsnorm_kernel.cuh"


namespace kernel
{
    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type)
    {
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            {
                return rmsnorm_kernel_cpu;
            }
            case base::DeviceType::DeviceCUDA:
            {
                return rmsnorm_kernel_cu;
            }
            default:
            {
                LOG(FATAL) << "Unsupported device type: " << device_type;
                return nullptr;
            }
        }
    }
}