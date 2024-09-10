#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>
#include "tensor/tensor.h"


namespace tensor
{
    template<typename T, typename Tp>
    static size_t reduce_dimensions(T begin, T end, Tp init)
    {
        if (begin >= end)
        {
            LOG(FATAL) << "Invalid range when reducing dimension.";
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    size_t Tensor::byte_size() const 
    {
        return this->size_ * tensor::data_type_size(this->dtype_);
    }

    size_t Tensor::size() const
    {
        return this->size_;
    }

    int32_t Tensor::dims_size() const
    {
        return static_cast<int32_t>(this->dims_.size());
    }

    base::DataType Tensor::dtype() const
    {
        return this->dtype_;
    }

    base::DeviceType Tensor::device_type() const
    {
        if (this->buffer_ == nullptr)
        {
            LOG(ERROR) << "The buffer is null pointer, the device type is unknown.";
            return base::DeviceType::DeviceUnknown;
        }
        return this->buffer_->device_type();
    }

    const std::vector<int32_t>& Tensor::dims() const
    {
        return this->dims_;
    }

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> alloc, bool need_realloc)
    {
        if (alloc == nullptr)
        {
            LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
            return false;
        }
        size_t byte_size = this->byte_size();
        if (this->buffer_ && byte_size <= this->buffer_->byte_size())
        {
            if (need_realloc == false) return true;
        }

        this->buffer_ = std::make_shared<base::Buffer>(byte_size,alloc, nullptr, false);
        if (this->buffer_ == nullptr)
        {
            LOG(ERROR) << "Failed to allocate memory for tensor.";
            return false;
        }
        return true;
    }

    void Tensor::init_buffer
    (
        std::shared_ptr<base::DeviceAllocator> alloc,
        base::DataType dtype,
        bool need_alloc,
        void* ptr
    )
    {
        // 外部引用
        if (need_alloc==false && alloc==nullptr)
        {
            std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(this->byte_size(), nullptr, ptr, true);
            this->buffer_ = buffer;
        }
        else
        {
            this->allocate(alloc, true);
        }
    }
    
    // kuiper的奇怪实现
    // void Tensor::initTensor
    // (
    //     bool need_alloc, 
    //     std::shared_ptr<base::DeviceAllocator> alloc,
    //     void* ptr
    // )
    // {
    //     if (need_alloc && alloc)
    //     {
    //         this->allocate(alloc);
    //     }
    //     else
    //     {
    //         if (ptr == nullptr)
    //         {
    //             LOG(FATAL) << "Data is nullptr and need_alloc is false";
    //         }
    //         else
    //         {
    //             CHECK(need_alloc == false) << "The need_alloc is is true when ptr parameter is not a null pointer.";
    //             this->init_buffer(alloc, this->dtype_, need_alloc, ptr);
    //         }
    //     }
    // }

    
    void Tensor::initTensor
    (
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void *ptr
    )
    {
        if (need_alloc)
        {
            this->allocate(alloc, true);
        }
        else if (ptr)
        {
            this->init_buffer(alloc, this->dtype_, need_alloc, ptr);
        }
        else return; // 初始化buffer为空，仅赋值dtype和dims，后续使用assign函数赋值buffer
    }

    Tensor::Tensor
    (
        base::DataType dtype,
        int32_t dim0,
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void* ptr
    )
    : dtype_(dtype)
    {
        this->dims_.push_back(dim0);
        this->size_ = dim0;
        this->initTensor(need_alloc, alloc, ptr);
    }


    Tensor::Tensor
    (
        base::DataType dtype,
        int32_t dim0,
        int32_t dim1,
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void* ptr
    )
    : dtype_(dtype)
    {
        this->dims_.push_back(dim0);
        this->dims_.push_back(dim1);
        this->size_ = dim0*dim1;
        this->initTensor(need_alloc, alloc, ptr);
    }

    Tensor::Tensor
    (
        base::DataType dtype,
        int32_t dim0,
        int32_t dim1,
        int32_t dim2,
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void* ptr
    )
    : dtype_(dtype)
    {
        this->dims_.push_back(dim0);
        this->dims_.push_back(dim1);
        this->dims_.push_back(dim2);
        this->size_ = dim0*dim1*dim2;
        this->initTensor(need_alloc, alloc, ptr);
    }

    Tensor::Tensor
    (
        base::DataType dtype,
        int32_t dim0,
        int32_t dim1,
        int32_t dim2,
        int32_t dim3,
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void* ptr
    )
    : dtype_(dtype)
    {
        this->dims_.push_back(dim0);
        this->dims_.push_back(dim1);
        this->dims_.push_back(dim2);
        this->dims_.push_back(dim3);
        this->size_ = dim0*dim1*dim2*dim3;
        this->initTensor(need_alloc, alloc, ptr);
    }

    Tensor::Tensor
    (
        base::DataType dtype,
        std::vector<int32_t> dims,
        bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc,
        void* ptr
    )
    : dtype_(dtype), dims_(dims)
    {
        this->size_ = tensor::reduce_dimensions(this->dims_.begin(), this->dims_.end(), 1);
        this->initTensor(need_alloc, alloc, ptr);
    }

    void Tensor::to_cpu()
    {
        CHECK(this->buffer_ != nullptr);
        const base::DeviceType device_type = this->device_type();
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            { 
                LOG(INFO) << "The tensor is already on CPU.";
                break;
            }
            case base::DeviceType::DeviceCUDA:
            {
                auto cpu_alloc = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCPU);
                auto cpu_buffer = std::make_shared<base::Buffer>(this->byte_size(), cpu_alloc, nullptr, false);
                cpu_alloc->memcpy(cpu_buffer->ptr(), this->buffer_->ptr(), this->byte_size(), base::MemcpyKind::MemcpyCUDA2CPU);
                this->buffer_ = cpu_buffer;
                break;
            }
            default:{ LOG(ERROR) << "Unsupported device type: " << base::DeviceTypeToString(device_type); }
        }
    }

    void Tensor::to_cuda(cudaStream_t stream)
    {
        CHECK(this->buffer_ != nullptr);
        const base::DeviceType device_type = this->device_type();
        switch (device_type)
        {
            case base::DeviceType::DeviceCPU:
            {
                auto cuda_alloc = base::DeviceAllocatorFactory::get_instance(base::DeviceType::DeviceCUDA);
                auto cuda_buffer = std::make_shared<base::Buffer>(this->byte_size(), cuda_alloc, nullptr, false);
                cuda_alloc->memcpy(cuda_buffer->ptr(), this->buffer_->ptr(), this->byte_size(), base::MemcpyKind::MemcpyCPU2CUDA, stream);
                this->buffer_ = cuda_buffer;
                break;
            }
            case base::DeviceType::DeviceCUDA:
            { 
                LOG(INFO) << "The tensor is already on CUDA.";
                break;
            }
            default:{ LOG(ERROR) << "Unsupported device type: " << base::DeviceTypeToString(device_type); }
        }
    }

    bool Tensor::is_empty() const
    {
        return this->size_ == 0 ||
               this->buffer_ == nullptr ||
               this->buffer_->ptr() == nullptr;
    }

    int32_t Tensor::get_dim(int32_t index) const
    {
        if (index <0 || index >= this->dims_.size())
        {
            LOG(FATAL) << "Invalid index: " << index << ", when getting dimension " << this->dims_.size();
        }
        return this->dims_[index];
    }

    void Tensor::reshape(const std::vector<int32_t>& dims)
    {
        size_t size = tensor::reduce_dimensions(dims.begin(), dims.end(), 1);
        if (this->buffer_ == nullptr)
        {
            this->size_ = size;
            this->dims_ = dims;
            return;
        }
        if (size > this->size_)
        {
            auto new_buffer = std::make_shared<base::Buffer>(this->byte_size(), this->buffer_->allocator(), nullptr, false);
            if (new_buffer->allocate() == false)
            {
                LOG(FATAL) << "Failed to allocate memory for tensor.";
            }
            new_buffer->copy_from(this->buffer_.get());
            this->buffer_ = new_buffer;
        }
        this->size_ = size;
        this->dims_ = dims;
    }

    std::vector<size_t> Tensor::strides() const
    {
        std::vector<size_t> strides;
        if (this->dims_.empty() == false)
        {
            for (int32_t i = 0; i < this->dims_.size() - 1; ++i)
            {
                strides.push_back(reduce_dimensions(this->dims_.begin() + i + 1, this->dims_.end(), 1));
            }
            strides.push_back(1);
        }
        return strides;
    }
    
    // 有点像赋值，应用场景是新建一个tensor，然后把原tensor的buffer赋值给新tensor
    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer)
    {
        if (buffer == nullptr)
        {
            LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
            return false;
        }
        if (this->buffer_ != nullptr)
        {
            if (this->device_type() != buffer->device_type())
            {
                LOG(ERROR) << "The device type of the tensor and the buffer are different.";
                return false;
            }
        }
        // 
        if (this->byte_size() > buffer->byte_size())
        {
            LOG(ERROR) << "The size of the tensor is larger than the size of the buffer.";
            return false;
        }

        this->buffer_ = buffer;
        return true;
    }

    // 重塑这个tensor，并将其buffer置为nullptr，等待下一次分配
    void Tensor::reset(base::DataType dtype, const std::vector<int32_t>& dims)
    {   
        this->dtype_ = dtype;
        this->dims_ = dims;
        this->size_ = tensor::reduce_dimensions(this->dims_.begin(), this->dims_.end(), 1);
        this->buffer_ = nullptr;
    }

    Tensor Tensor::clone() const
    {
        Tensor new_tensor(this->dtype_, this->dims_);
        if (this->buffer_ == nullptr) return new_tensor;
        else
        {
            new_tensor.buffer_ = std::make_shared<base::Buffer>(this->byte_size(), this->buffer_->allocator(), nullptr, false);
            new_tensor.buffer_->copy_from(this->buffer_.get());
            return new_tensor;
        }
    }

} // namespace tensor