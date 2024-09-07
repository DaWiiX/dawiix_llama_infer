#include <cuda_runtime_api.h>
#include "../../include/base/alloc.h"

namespace base {
    CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::DeviceCUDA)
    {}

    void* CUDADeviceAllocator::allocate(size_t byte_size) const
    {
        int device_id = -1;
        cudaError_t status = cudaGetDevice(&device_id);
        CHECK(status == cudaSuccess) << "Failed to get device id";

        // allocate big mem buffer
        if (byte_size > 500 * 1024 * 1024)
        {
            auto& big_buffers = this->big_buffers_map_[device_id];
            int sel_id = -1;

            if (this->big_no_busy_cnt_[device_id] > byte_size)
            {
                for(int i = 0; i < big_buffers.size(); ++i)
                {
                    if 
                    (
                        big_buffers[i].byte_size >= byte_size && // 起码要大于我们申请的大小
                        big_buffers[i].busy == false && // 这个地方是空闲的
                        big_buffers[i].byte_size - byte_size < 1024 * 1024 // 这个地方不可以太大，不然太浪费了，我们需要重新申请
                    )
                    {
                        if 
                        (
                            sel_id == -1 || // 还没有选中的话就先选一个
                            big_buffers[i].byte_size < big_buffers[sel_id].byte_size // 往后面找合适的当中最小的，最大化利用空间
                        )
                        {
                            sel_id = i;
                        }
                    }
                }
            }
            
            // 找到了我们就返回
            if (sel_id != -1)
            {
                big_buffers[sel_id].busy = true;
                this->big_no_busy_cnt_[device_id] -= byte_size;
                return big_buffers[sel_id].data;
            }

            // 没找到就申请新的
            void* ptr = nullptr;
            status = cudaMalloc(&ptr, byte_size);
            if (status != cudaSuccess)
            {
                char buf[256];
                snprintf
                (
                    buf, 256,
                    "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on  device.",
                    byte_size >> 20
                );
                LOG(ERROR) << buf;
                return nullptr;
            }

            // 更新big_buffers_map_
            big_buffers.emplace_back(ptr, byte_size, true);
            return ptr;
        }
        
        // allocate small mem buffer
        auto& cuda_buffer = this->cuda_buffers_map_[device_id];
        for (int i = 0; i < cuda_buffer.size(); ++i)
        {
            if
            (
                cuda_buffer[i].byte_size >= byte_size && // 起码要大于我们申请的大小
                cuda_buffer[i].busy == false // 这个地方是空闲的
            )
            {
                cuda_buffer[i].busy = true;
                this->no_busy_cnt_[i] -= cuda_buffer[i].byte_size;
                return cuda_buffer[i].data;
            }
        }
        // 没找到就申请新的
        void* ptr = nullptr;
        status = cudaMalloc(&ptr, byte_size);
        if (status != cudaSuccess)
        {
            char buf[256];
            snprintf
            (
                buf, 256,
                "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on  device.",
                byte_size >> 20
            );
            LOG(ERROR) << buf;
            return nullptr;
        }
        // 更新cuda_buffers_map_
        cuda_buffer.emplace_back(ptr, byte_size, true);
        return ptr;
        
    }

    void CUDADeviceAllocator::release(void* ptr) const
    {
        if (ptr == nullptr)
        {
            LOG(WARNING) << "you are trying to release a nullptr pointer";
            return;
        }
        if (this->cuda_buffers_map_.empty())
        {
            LOG(WARNING) << "No CUDA buffer to release";
            return;
        }

        // 遍历每个设备，如果该设备中的小内存占用超过某个阈值，我们将尝试释放空闲区
        cudaError_t status = cudaSuccess;
        for (auto& it : this->cuda_buffers_map_)
        {
            if (this->no_busy_cnt_[it.first] > 1024*1024*1024)
            {
                status = cudaSetDevice(it.first);
                auto& cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> temp;
                for (int i = 0; i < cuda_buffers.size(); ++i)
                {
                    // 找到空闲区并尝试释放
                    if (cuda_buffers[i].busy == false)
                    {
                        status = cudaFree(cuda_buffers[i].data);
                        CHECK(status == cudaSuccess) << "Error: CUDA error when release memory on device " << it.first;
                    }
                    // 如果不是空闲区就还装回去
                    else temp.push_back(cuda_buffers[i]);
                }
            }
        }

        // 遍历每个设备，如果该设备中的大内存占用超过某个阈值，我们将尝试释放空闲区
        for (auto& it : this->big_buffers_map_)
        {
            size_t threshold = 8ULL*1024*1024*1024;
            if (this->big_no_busy_cnt_[it.first] > threshold)
            {
                status = cudaSetDevice(it.first);
                auto& cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> temp;
                for (int i = 0; i < cuda_buffers.size(); ++i)
                {
                    // 找到空闲区并尝试释放
                    if (cuda_buffers[i].busy == false)
                    {
                        status = cudaFree(cuda_buffers[i].data);
                        CHECK(status == cudaSuccess) << "Error: CUDA error when release memory on device " << it.first;
                    }
                    // 如果不是空闲区就还装回去
                    else temp.push_back(cuda_buffers[i]);
                }
            }
        }

        // 表面删除ptr，但实际上没有删除，只是标记为空闲，等待下次分配
        for (auto& it : this->cuda_buffers_map_)
        {
            auto& cuda_buffers = it.second;
            for (int i = 0; i < cuda_buffers.size(); ++i)
            {
                if (cuda_buffers[i].data == ptr)
                {
                    cuda_buffers[i].busy = false;
                    this->no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                    return;
                }
            }
        }

        // 表面删除ptr，但实际上没有删除，只是标记为空闲，等待下次分配
        for (auto& it : this->big_buffers_map_)
        {
            auto& cuda_buffers = it.second;
            for (int i = 0; i < cuda_buffers.size(); ++i)
            {
                if (cuda_buffers[i].data == ptr)
                {
                    cuda_buffers[i].busy = false;
                    this->big_no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                    return;
                }
            }
        }

        LOG(WARNING) << "you are trying to release a pointer that is not allocated by this allocator";
        status = cudaFree(ptr);
        CHECK(status == cudaSuccess) << "Error: CUDA error when release memory";
        return;
    }

    std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;


} // namespace base