#include <glog/logging.h>
#include "base/buffer.h"


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
            this->device_type_ = this->allocator_->device_type();
            this->use_external_ = false;
            this->ptr_ = this->allocator_->allocate(byte_size);
        }
    }

    Buffer::~Buffer()
    {
        if 
        (
            this->use_external_ == false     &&
            this->ptr_          != nullptr   &&
            this->allocator_
        )
        {
            // LOG(INFO) << "Releasing buffer of size " << this->byte_size_ << " bytes";
            this->allocator_->release(this->ptr_);
            this->ptr_ = nullptr;
        }
    }

    bool Buffer::allocate()
    {
        if (this->allocator_ && this->byte_size_ != 0)
        {
            this->use_external_ = false;
            ptr_ = this->allocator_->allocate(this->byte_size_);
            if (ptr_ == nullptr)
            {
                LOG(ERROR) << "Failed to allocate memory for buffer of size " << this->byte_size_;
                return false;
            }
            else return true;
        }
        else
        {
            LOG(ERROR) << "Cannot allocate memory for buffer without allocator or byte size " << this->byte_size_ << " bytes";
            return false;
        }
    }

    void Buffer::copy_from(const Buffer& buffer) const
    {
        CHECK(this->allocator_ != nullptr);
        CHECK(buffer.ptr_ != nullptr);

        size_t bype_size = this->byte_size_ < buffer.byte_size_? this->byte_size_ : buffer.byte_size_;
        const DeviceType buffer_device = buffer.device_type();
        const DeviceType this_device = this->device_type();
        CHECK(buffer_device != DeviceType::DeviceUnknown && this_device != DeviceType::DeviceUnknown);

        if 
        (
            buffer_device == DeviceType::DeviceCPU &&
            this_device   == DeviceType::DeviceCPU
        ) return this->allocator_->memcpy(this->ptr_, buffer.ptr_, bype_size, MemcpyKind::MemcpyCPU2CPU);

        else if 
        (
            buffer_device == DeviceType::DeviceCPU &&
            this_device   == DeviceType::DeviceCUDA
        ) return this->allocator_->memcpy(this->ptr_, buffer.ptr_, bype_size, MemcpyKind::MemcpyCPU2CUDA);
        
        else if 
        (
            buffer_device == DeviceType::DeviceCUDA &&
            this_device   == DeviceType::DeviceCPU
        ) return this->allocator_->memcpy(this->ptr_, buffer.ptr_, bype_size, MemcpyKind::MemcpyCUDA2CPU);
        
        else if 
        (
            buffer_device == DeviceType::DeviceCUDA &&
            this_device   == DeviceType::DeviceCUDA
        ) return this->allocator_->memcpy(this->ptr_, buffer.ptr_, bype_size, MemcpyKind::MemcpyCUDA2CUDA);
        
        else
        {
            LOG(ERROR) << "Unsupported device type for copy operation";
            return;
        }
    }

    void Buffer::copy_from(const Buffer* buffer) const 
    {
        CHECK(this->allocator_ != nullptr);
        CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

        size_t bype_size = this->byte_size_ < buffer->byte_size_? this->byte_size_ : buffer->byte_size_;
        const DeviceType buffer_device = buffer->device_type();
        const DeviceType this_device = this->device_type();
        CHECK(buffer_device != DeviceType::DeviceUnknown && this_device != DeviceType::DeviceUnknown);

        if 
        (
            buffer_device == DeviceType::DeviceCPU &&
            this_device   == DeviceType::DeviceCPU
        ) return this->allocator_->memcpy(this->ptr_, buffer->ptr_, bype_size, MemcpyKind::MemcpyCPU2CPU);

        else if 
        (
            buffer_device == DeviceType::DeviceCPU &&
            this_device   == DeviceType::DeviceCUDA
        ) return this->allocator_->memcpy(this->ptr_, buffer->ptr_, bype_size, MemcpyKind::MemcpyCPU2CUDA);
        
        else if 
        (
            buffer_device == DeviceType::DeviceCUDA &&
            this_device   == DeviceType::DeviceCPU
        ) return this->allocator_->memcpy(this->ptr_, buffer->ptr_, bype_size, MemcpyKind::MemcpyCUDA2CPU);
        
        else if 
        (
            buffer_device == DeviceType::DeviceCUDA &&
            this_device   == DeviceType::DeviceCUDA
        ) return this->allocator_->memcpy(this->ptr_, buffer->ptr_, bype_size, MemcpyKind::MemcpyCUDA2CUDA);
        
        else
        {
            LOG(ERROR) << "Unsupported device type for copy operation";
            return;
        }
    }

    void* Buffer::ptr() { return this->ptr_; }

    const void* Buffer::ptr() const { return this->ptr_; }

    size_t Buffer::byte_size() const { return this->byte_size_; }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const { return this->allocator_; }

    DeviceType Buffer::device_type() const { return this->device_type_; }

    void Buffer::set_device_type(const DeviceType device_type) { this->device_type_ = device_type; }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }

    bool Buffer::is_external() const { return this->use_external_; }

} // namespace base