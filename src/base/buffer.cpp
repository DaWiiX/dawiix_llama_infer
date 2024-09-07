#include "../../include/base/buffer.h"
#include <glog/logging.h>

namespace base
{
    Buffer::Buffer
    (
        size_t byte_size,
        std::shared_ptr<DeviceAllocator> allocator,
        void* ptr,
        bool use_external
    )
    :   byte_size_(byte_size),
        allocator_(allocator),
        ptr_(ptr),
        use_external_(use_external)
    {
        if (this->ptr_ == nullptr && this->allocator_ != nullptr)
        {
            this->use_external_ = false;
            this->ptr_ = this->allocator_->allocate(this->byte_size_)
        }
    }

    Buffer::~Buffer()
    {
        if 
        (
            this->use_external_ == false &&
            this->ptr_        != nullptr &&
            this->allocator_
        )
        {
            this->allocator_->release(this->ptr_);
            this->ptr_ = nullptr;
        }
    }

    void* Buffer::ptr()
    {
        return this->ptr_;
    }

    const void* Buffer::ptr() const
    {
        return this->ptr_;
    }

    const size_t Buffer::byte_size() const
    {
        return this->byte_size_;
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const
    {
        return this->allocator_;
    }

    const DeviceType Buffer::device_type() const
    {
        return this->device_type_;
    }

    void Buffer::set_device_type(const DeviceType device_type)
    {
        this->device_type_ = device_type;
    }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this()
    {
        return shared_from_this();
    }

    bool Buffer::is_external() const
    {
        return this->use_external_;
    }

} // namespace base