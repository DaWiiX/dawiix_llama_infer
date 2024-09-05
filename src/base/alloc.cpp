#include "../../include/base/alloc.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace base 
{
    void DeviceAllocator::memcpy
    (
        const void* src_ptr,
        void* dest_ptr,
        size_t byte_size,
        MemcpyKind memcpy_kind,
        void* stream,
        bool need_sync
    ) const
    {
        // precheck
        CHECK_NE(src_ptr, nullptr);
        CHECK_NE(dest_ptr, nullptr);
        CHECK_NE(byte_size, 0);

        cudaStream_t stream_ = nullptr;
        if (stream) stream_ = static_cast<CUstream_st*>(stream);

        switch (memcpy_kind) 
        {
            case MemcpyKind::MemcpyCPU2CPU:
                std::memcpy(dest_ptr, src_ptr, byte_size);
                break;
            
            case MemcpyKind::MemcpyCPU2CUDA:
                if (stream_) cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
                else cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
                break;
            
            case MemcpyKind::MemcpyCUDA2CPU:
                if (stream_) cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
                else cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
                break;

            case MemcpyKind::MemcpyCUDA2CUDA:
                if (stream_) cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
                else cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
                break;

            default:
                LOG(FATAL) << "Invalid memcpy kind: " << static_cast<int>(memcpy_kind);
                break;
        }

        if (need_sync) cudaDeviceSynchronize();
    }

    void DeviceAllocator::memset_zero
    (
        void* ptr,
        size_t byte_size,
        void* stream,
        bool need_sync
    )
    {
        // precheck
        CHECK_NE(this->device_type_, DeviceType::DeviceUnknown);
        cudaStream_t stream_ = nullptr;
        if (stream) stream_ = static_cast<CUstream_st*>(stream);
        switch (this->device_type_) 
        {
            case DeviceType::DeviceCPU:
                std::memset(ptr, 0, byte_size);
                break;
            
            case DeviceType::DeviceCUDA:
                if (stream_) cudaMemsetAsync(ptr, 0, byte_size, stream_);
                else cudaMemset(ptr, 0, byte_size);
                break;

            default:
                LOG(FATAL) << "Invalid device type: " << DeviceTypeToString(this->device_type_);
                break;
        }
    }


}// namespace base