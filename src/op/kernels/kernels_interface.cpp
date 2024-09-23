#include "base/base.h"
#include "op/kernels/kernels_interface.h"

#include "op/kernels/cpu/add_kernel.h"
#include "op/kernels/cuda/add_kernel.cuh"
#include "op/kernels/cpu/matmul_kernel.h"
#include "op/kernels/cuda/matmul_kernel.cuh"
#include "op/kernels/cpu/emb_kernel.h"
#include "op/kernels/cuda/emb_kernel.cuh"
#include "op/kernels/cpu/rmsnorm_kernel.h"
#include "op/kernels/cuda/rmsnorm_kernel.cuh"



namespace kernel
{
    AddKernel get_add_kernel(base::DeviceType device_type)
    {
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            {
                return add_kernel_cpu;
            }
            case base::DeviceType::DeviceCUDA:
            {
                return add_kernel_cu;
            }
            default:
            {
                LOG(FATAL) << "Unsupported device type: " << device_type;
                return nullptr;
            }
        }
    }

    MatmulKernel get_matmul_kernel(base::DeviceType device_type)
    {
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            {
                return matmul_kernel_cpu;
            }
            case base::DeviceType::DeviceCUDA:
            {
                return matmul_kernel_cu;
            }
            default:
            {
                LOG(FATAL) << "Unsupported device type: " << device_type;
                return nullptr;
            }
        }
    }

    MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type)
    {
        switch (device_type)
        {
            case base::DeviceType::DeviceCUDA:
            {
                return matmul_kernel_cu_qint8;
            }
            default:
            {
                LOG(FATAL) << "Unsupported device type: " << device_type;
                return nullptr;
            }
        }
    }

    EmbeddingKernel get_emb_kernel(base::DeviceType device_type)
    {
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            {
                return emb_kernel_cpu;
            }
            case base::DeviceType::DeviceCUDA:
            {
                return emb_kernel_cu;
            }
            default:
            {
                LOG(FATAL) << "Unsupported device type: " << device_type;
                return nullptr;
            }
        }
    }

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