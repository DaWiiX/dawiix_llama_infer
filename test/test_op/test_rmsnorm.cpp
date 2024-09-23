#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

using namespace kernel;
TEST(RMSNormTest, RMSNormNOSTREAM)
{
    auto alloc_cu = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCUDA);
    auto alloc_cpu = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCPU);

    int32_t size = 32 * 15;

    tensor::Tensor in_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
        // square_sum += pow(dist(mt), 2);

        // in_cpu.index<float>(i) = static_cast<float>(i);
        // wei_cpu.index<float>(i) = static_cast<float>(i);
        // square_sum += pow(static_cast<float>(i), 2);
    }

    float rsquare_mean = sqrt(square_sum / static_cast<float>(size) + 1e-5f);
    rsquare_mean = 1.0f / rsquare_mean;
    std::vector<float> gt(size, 0);
    for (int i = 0; i < size; ++i)
    {
        gt[i] = i * i * rsquare_mean;
    }

    tensor::Tensor in_cu = in_cpu.clone();
    tensor::Tensor wei_cu = wei_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in_cu.to_cuda(nullptr);
    wei_cu.to_cuda(nullptr);
    out_cu.to_cuda(nullptr);

    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
    out_cu.to_cpu();

    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                            nullptr);

    for (int i = 0; i < size; ++i)
    {
        // EXPECT_NEAR(out_cu.index<float>(i), gt[i], 1e-5f);
        // EXPECT_NEAR(out_cpu.index<float>(i), gt[i], 1e-5f);
        EXPECT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
}

TEST(RMSNormTest, RMSNormSTREAM1)
{
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32;

    tensor::Tensor in_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; ++i)
    {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in_cu = in_cpu.clone();
    tensor::Tensor wei_cu = wei_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in_cu.to_cuda(nullptr);
    wei_cu.to_cuda(nullptr);
    out_cu.to_cuda(nullptr);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(in_cu, wei_cu, out_cu,
                                                              stream);
    out_cu.to_cpu();

    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                             nullptr);

    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}

TEST(RMSNormTest, RMSNormSTREAM2)
{
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151 * 15;

    tensor::Tensor in_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::DataTypeFp32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; ++i)
    {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in_cu = in_cpu.clone();
    tensor::Tensor wei_cu = wei_cpu.clone();
    tensor::Tensor out_cu = out_cpu.clone();
    in_cu.to_cuda(nullptr);
    wei_cu.to_cuda(nullptr);
    out_cu.to_cuda(nullptr);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCUDA)(in_cu, wei_cu, out_cu,
                                                              stream);
    out_cu.to_cpu();

    kernel::get_rmsnorm_kernel(base::DeviceType::DeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                             nullptr);

    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}