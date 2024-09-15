#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <vector>

#include "../utils.cuh"
#include "base/buffer.h"


TEST(TensorTest, ConstructorCPU)
{
    base::DataType dtype = base::DataType::DataTypeFp32;
    std::vector<int32_t> dims = {2, 3, 4};
    bool need_alloc = true;
    std::shared_ptr<base::DeviceAllocator> alloc = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCPU);

    tensor::Tensor test_tensor(dtype, dims, need_alloc, alloc);

    EXPECT_EQ(test_tensor.size(), 24);
    EXPECT_EQ(test_tensor.dtype(), dtype);
    EXPECT_EQ(test_tensor.byte_size(), 24 * tensor::data_type_size(dtype));
    EXPECT_EQ(test_tensor.device_type(), base::DeviceType::DeviceCPU);
    EXPECT_EQ(test_tensor.dims(), dims);

    for (int32_t i = 0; i <= 2; i++)
    {
        EXPECT_EQ(test_tensor.get_dim(i), dims[i]);
    }

    EXPECT_EQ(test_tensor.is_empty(), false);
}

TEST(TensorTest, ConstructorCUDA)
{
    base::DataType dtype = base::DataType::DataTypeFp32;
    std::vector<int32_t> dims = {2, 3, 4};
    bool need_alloc = true;
    std::shared_ptr<base::DeviceAllocator> alloc = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCUDA);

    tensor::Tensor test_tensor(dtype, dims, need_alloc, alloc);

    EXPECT_EQ(test_tensor.size(), 24);
    EXPECT_EQ(test_tensor.dtype(), dtype);
    EXPECT_EQ(test_tensor.byte_size(), 24 * tensor::data_type_size(dtype));
    EXPECT_EQ(test_tensor.device_type(), base::DeviceType::DeviceCUDA);
    EXPECT_EQ(test_tensor.dims(), dims);

    for (int32_t i = 0; i <= 2; i++)
    {
        EXPECT_EQ(test_tensor.get_dim(i), dims[i]);
    }

    EXPECT_EQ(test_tensor.is_empty(), false);
}

TEST(TensorTest, To_CPU)
{
    // using namespace base;
    auto alloc_cu = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCUDA);

    std::vector<int32_t> dims = {2, 3, 4};
    base::DataType dtype = base::DataType::DataTypeFp32;
    tensor::Tensor t1_cu(dtype, dims, true, alloc_cu);
    EXPECT_EQ(t1_cu.is_empty(), false);
    set_value_cu(t1_cu.ptr<float>(), 2*3*4, 1.0f);

    t1_cu.to_cpu();
    EXPECT_EQ(t1_cu.device_type(), base::DeviceType::DeviceCPU);
    float *cpu_ptr = t1_cu.ptr<float>();
    for (int i = 0; i < 2 * 3 * 4; ++i)
    {
        EXPECT_EQ(*(cpu_ptr + i), 1.f);
    }
}

TEST(TensorTest, To_CUDA)
{
    using namespace base;
    auto alloc_cpu = DeviceAllocatorFactory::get_instance(DeviceType::DeviceCPU);
    tensor::Tensor t1_cpu(DataType::DataTypeFp32, 32, 32, true, alloc_cpu);
    EXPECT_EQ(t1_cpu.is_empty(), false);
    float* p1 = t1_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i)
    {
        *(p1 + i) = 1.f;
    }

    t1_cpu.to_cuda();
    EXPECT_EQ(t1_cpu.device_type(), DeviceType::DeviceCUDA);
    float* p2 = new float[32 * 32];
    cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(p2 + i), 1.f);
    }
    delete[] p2;
}

TEST(TensorTest, CUDA_Clone)
{
    using namespace base;
    auto alloc_cu = DeviceAllocatorFactory::get_instance(DeviceType::DeviceCUDA);
    tensor::Tensor t1_cu(DataType::DataTypeFp32, 32, 32, true, alloc_cu);
    EXPECT_EQ(t1_cu.is_empty(), false);
    set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.0f);

    tensor::Tensor t2_cu = t1_cu.clone();
    EXPECT_EQ(t2_cu.device_type(), DeviceType::DeviceCUDA);
    EXPECT_EQ(t2_cu.size(), 32 * 32);
    EXPECT_EQ(t2_cu.byte_size(), 32 * 32 * tensor::data_type_size(DataType::DataTypeFp32));
    EXPECT_EQ(t2_cu.dtype(), DataType::DataTypeFp32);

    float* p1 = new float[32 * 32];
    float* p2 = new float[32 * 32];
    cudaMemcpy(p1, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(p1 + i), *(p2 + i));
        EXPECT_EQ(*(p1 + i), 1.f);
        EXPECT_EQ(*(p2 + i), 1.f);
    }

    t2_cu.to_cpu();
    EXPECT_EQ(t2_cu.device_type(), DeviceType::DeviceCPU);
    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(p2 + i), *(t2_cu.ptr<float>() + i));
        EXPECT_EQ(*(p1 + i), *(t2_cu.ptr<float>() + i));

    }
    delete[] p1;
    delete[] p2;
}

TEST(TensorTest, CPU_Clone)
{
    using namespace base;
    auto alloc_cpu = DeviceAllocatorFactory::get_instance(DeviceType::DeviceCPU);
    tensor::Tensor t1_cpu(DataType::DataTypeFp32, 32, 32, true, alloc_cpu);
    EXPECT_EQ(t1_cpu.is_empty(), false);
    float* p1 = t1_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i)
    {
        *(p1 + i) = 1.f;
    }

    tensor::Tensor t2_cpu = t1_cpu.clone();
    EXPECT_EQ(t2_cpu.device_type(), DeviceType::DeviceCPU);
    EXPECT_EQ(t2_cpu.size(), 32 * 32);
    EXPECT_EQ(t2_cpu.byte_size(), 32 * 32 * tensor::data_type_size(DataType::DataTypeFp32));
    EXPECT_EQ(t2_cpu.dtype(), DataType::DataTypeFp32);

    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(p1 + i), *(t2_cpu.ptr<float>() + i));
        EXPECT_EQ(*(p1 + i), 1.f);
        EXPECT_EQ(*(t2_cpu.ptr<float>() + i), 1.f);
    }


    t2_cpu.to_cuda();
    EXPECT_EQ(t2_cpu.device_type(), DeviceType::DeviceCUDA);
    float* p2 = new float[32 * 32];
    cudaMemcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(p2 + i), 1.f);
        EXPECT_EQ(*(p1 + i), *(p2 + i));
    }
    delete[] p2;
}

TEST(TensorTest, Assign)
{
    using namespace base;
    auto alloc_cpu = DeviceAllocatorFactory::get_instance(DeviceType::DeviceCPU);
    tensor::Tensor t1_cpu(DataType::DataTypeFp32, 32, 32, true, alloc_cpu);
    EXPECT_EQ(t1_cpu.is_empty(), false);
    float* p1 = t1_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i)
    {
        *(p1 + i) = 1.f;
    }

    tensor::Tensor t2_cpu(DataType::DataTypeFp32, 32, 32);
    auto buffer = std::make_shared<Buffer>(32*32*sizeof(float), alloc_cpu,nullptr,false);
    LOG(INFO) << "buffer count " << buffer.use_count();
    auto p2 = buffer->get_shared_from_this();
    LOG(INFO) << "buffer count " << buffer.use_count();
    for (int i = 0; i < 32 * 32; ++i)
    {
        *(reinterpret_cast<float*>((p2.get())->ptr()) + i) = 1.f;
    }

    t2_cpu.assign(buffer);
    LOG(INFO) << "buffer count " << buffer.use_count();
    EXPECT_EQ(t2_cpu.device_type(), DeviceType::DeviceCPU);
    EXPECT_EQ(t2_cpu.size(), 32 * 32);
    EXPECT_EQ(t2_cpu.byte_size(), 32 * 32 * tensor::data_type_size(DataType::DataTypeFp32));
    EXPECT_EQ(t2_cpu.dtype(), DataType::DataTypeFp32);
    for (int i = 0; i < 32 * 32; ++i)
    {
        EXPECT_EQ(*(reinterpret_cast<float *>((p2.get())->ptr()) + i), 1.f);
        EXPECT_EQ(*(t2_cpu.ptr<float>() + i), *(p1 + i));
    }
    // free(p2);
    LOG(INFO) << "buffer count " << buffer.use_count();
}