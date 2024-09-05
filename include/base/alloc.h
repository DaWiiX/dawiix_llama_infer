#ifndef INCLUDE_BASE_ALLOC_H_
#define INCLUDE_BASE_ALLOC_H_
#include "base.h"

namespace base 
{
    enum class MemcpyKind
    {
        MemcpyCPU2CPU = 0,
        MemcpyCPU2CUDA = 1,
        MemcpyCUDA2CPU = 2,
        MemcpyCUDA2CUDA = 3,
    };

    class DeviceAllocator 
    {
    public:
        explicit DeviceAllocator(DeviceType device_type)
        : device_type_(device_type)
        {}

        virtual DeviceType device_type() const {return this->device_type_;}

        virtual void* allocate(size_t byte_size) const = 0;

        virtual void release(void* ptr) const = 0;

        virtual void memcpy
        (
            const void* src_ptr,
            void* dest_ptr,
            size_t byte_size,
            MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPU2CPU,
            void* stream = nullptr,
            bool need_sync = false
        ) const;

        virtual void memset_zero
        (
            void* ptr,
            size_t byte_size,
            void* stream,
            bool need_sync = false
        );

    private:
        DeviceType device_type_ = DeviceType::DeviceUnknown;
    };

    class CPUDeviceAllocator : public DeviceAllocator
    {
        public:
            explicit CPUDeviceAllocator();

            void* allocate(size_t byte_size) const override;

            void release(void* ptr) const override;
    };

    struct CudaMemoryBuffer 
    {
        void* data;
        size_t byte_size;
        bool busy;

        CudaMemoryBuffer() = default;

        CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy)
        {}
    };
    
} // namespace base

#endif // INCLUDE_BASE_ALLOC_H_