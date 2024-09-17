#ifndef INCLUDE_TENSOR_H_
#define INCLUDE_TENSOR_H_
#include <driver_types.h>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"


namespace tensor
{
    class Tensor
    {
        private:
            size_t size_ = 0;
            std::vector<int32_t> dims_;
            std::shared_ptr<base::Buffer> buffer_;
            base::DataType dtype_ = base::DataType::DataTypeUnknown;
            

        public:
            explicit Tensor() = default;
            explicit Tensor
            (
                base::DataType dtype,
                int32_t dim0,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
            );

            explicit Tensor
            (
                base::DataType dtype,
                int32_t dim0,
                int32_t dim1,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
            );

            explicit Tensor
            (
                base::DataType dtype,
                int32_t dim0,
                int32_t dim1,
                int32_t dim2,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
            );

            explicit Tensor
            (
                base::DataType dtype,
                int32_t dim0,
                int32_t dim1,
                int32_t dim2,
                int32_t dim3,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
            );

            explicit Tensor
            (
                base::DataType dtype,
                std::vector<int32_t> dims,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
            );

            bool allocate(std::shared_ptr<base::DeviceAllocator> alloc, bool need_realloc=false);

            void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType dtype, bool need_alloc, void* ptr);

            void initTensor(bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr);

            size_t size() const;

            size_t byte_size() const;

            int32_t dims_size() const;

            base::DataType dtype() const;

            base::DeviceType device_type() const;

            void to_cpu();

            void to_cuda(cudaStream_t stream=nullptr);

            bool is_empty() const;

            int32_t get_dim(int32_t index) const;

            const std::vector<int32_t>& dims() const;

            void reshape(const std::vector<int32_t>& dims);

            std::vector<size_t> strides() const;

            bool assign(std::shared_ptr<base::Buffer> buffer);

            void set_device_type(base::DeviceType device_type) const;

            void reset(base::DataType dtype, const std::vector<int32_t>& dims);

            tensor::Tensor clone() const;

            template<typename T>
            T* ptr();

            template<typename T>
            const T* ptr() const;

            template<typename T>
            T* ptr(int64_t index);

            template<typename T>
            const T* ptr(int64_t index) const;

            template<typename T>
            T& index(int64_t offset);

            template<typename T>
            const T& index(int64_t offset) const;
    };

    template<typename T>
    T* Tensor::ptr()
    {
        if (this->buffer_ == nullptr) return nullptr;
        return reinterpret_cast<T*>(this->buffer_->ptr());
    }

    template<typename T>
    const T* Tensor::ptr() const
    {
        if (this->buffer_ == nullptr) return nullptr;
        return reinterpret_cast<const T*>(this->buffer_->ptr());
    }

    template<typename T>
    T* Tensor::ptr(int64_t index)
    {
        CHECK(this->buffer_ != nullptr) << "The data area buffer of this tensor is empty.";
        CHECK(this->buffer_->ptr() != nullptr) << "The data area buffer of this tensor is pointing to nullptr.";
        CHECK(index >= 0 && index < this->size_) << "The index is out of range.";

        return reinterpret_cast<T*>(this->buffer_->ptr()) + index;
    }

    template<typename T>
    const T* Tensor::ptr(int64_t index) const
    {
        CHECK(this->buffer_ != nullptr) << "The data area buffer of this tensor is empty.";
        CHECK(this->buffer_->ptr() != nullptr) << "The data area buffer of this tensor is pointing to nullptr.";
        CHECK(index >= 0 && index < this->size_) << "The index is out of range.";

        return reinterpret_cast<const T*>(this->buffer_->ptr()) + index;
    }

    template<typename T>
    T& Tensor::index(int64_t offset)
    {
        CHECK(offset >= 0 && offset < this->size_) << "The offset is out of range.";
        T &val = *(reinterpret_cast<T *>(this->buffer_->ptr()) + offset);
        return val;
    }

    template<typename T>
    const T& Tensor::index(int64_t offset) const
    {
        CHECK(offset >= 0 && offset < this->size_) << "The offset is out of range.";
        const T &val = *(reinterpret_cast<T *>(this->buffer_->ptr()) + offset);
        return val;
    }

    // inline static size_t data_type_size(base::DataType dtype)
    // {
    //     switch (dtype)
    //     {
    //     case base::DataType::DataTypeFp32:
    //         return 4;

    //     case base::DataType::DataTypeInt8:
    //         return 1;

    //     case base::DataType::DataTypeInt32:
    //         return 4;

    //     default:
    //         LOG(FATAL) << "Unsupported data type: " << static_cast<int>(dtype);
    //         return 0;
    //     }
    // }

    template <typename T, typename Tp>
    size_t reduce_dimensions(T begin, T end, Tp init);

}  // namespace tensor

#endif  // INCLUDE_TENSOR_H_