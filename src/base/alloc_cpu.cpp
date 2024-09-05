#include <glog/logging.h>
#include <cstdlib>
#include "../../include/base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define HAVE_USE_POSIX_MEMALIGN
#endif

namespace base {
    CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::DeviceCPU)
    {}

    void* CPUDeviceAllocator::allocate(size_t byte_size) const
    {
        if (byte_size == 0) return nullptr;

        #ifdef HAVE_USE_POSIX_MEMALIGN
        void* data = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
        int status = posix_memalign(
            (void**)&data,
            ((alignment < sizeof(void*)) ? sizeof(void*) : alignment),
            byte_size
        );
        if (status!= 0) {
            LOG(ERROR) << "Failed to allocate " << byte_size << " bytes of CPU memory with posix_memalign";
            return nullptr;
        }
        return data;

        #else
        return std::malloc(byte_size);

        #endif
    }

    void CPUDeviceAllocator::release(void* ptr) const
    {
        if (ptr == nullptr) return;
        else free(ptr);
    }

    std::shared_ptr<CPUDeviceAllocator> CUPDeviceAllocatorFactory::instance = nullptr;
} // namespace base