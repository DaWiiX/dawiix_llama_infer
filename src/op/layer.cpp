#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>
#include <base/cuda_config.h>
#include "op/layer.h"


namespace op
{
    // BaseLayer
    BaseLayer::BaseLayer
    (
        base::DeviceType device_type,
        LayerType layer_type,
        base::DataType dtype,
        std::string layer_name
    )
    : device_type_(device_type),
      layer_type_(layer_type),
      dtype_(dtype),
      layer_name_(std::move(layer_name))
    {}

    base::DataType BaseLayer::dtype() const { return this->dtype_; }

    base::DeviceType BaseLayer::device_type() const { return this->device_type_; }

    void BaseLayer::set_device_type(base::DeviceType device_type) { this->device_type_ = device_type; }

    LayerType BaseLayer::layer_type() const { return this->layer_type_; }

    const std::string& BaseLayer::get_layer_name() const { return this->layer_name_; }

    void BaseLayer::set_layer_name(const std::string& layer_name) { this->layer_name_ = layer_name; }

    base::Status BaseLayer::set_weight(int32_t index, const tensor::Tensor& weight) { return base::error::FunctionNotImplement(); }

    base::Status BaseLayer::set_weight
    (
        int32_t index,
        const std::vector<int32_t>& dims,
        const void *weight_ptr,
        base::DeviceType device_type
    )
    { return base::error::FunctionNotImplement(); }
    // End of BaseLayer

    // Layer
    Layer::Layer
    (
        base::DeviceType device_type,
        LayerType layer_type,
        std::string layer_name
    )
    : BaseLayer(device_type, layer_type, base::DataType::DataTypeFp32, layer_name)
    {}

    base::Status Layer::init()
    {
        return base::error::Success();
    }

    void Layer::set_input(int32_t index, const tensor::Tensor& input) 
    {
        CHECK(index >= 0);
        CHECK(index < this->inputs_.size());
        this->inputs_.at(index) = input;
    }

    void Layer::set_output(int32_t index, const tensor::Tensor& output)
    {
        CHECK(index >= 0);
        CHECK(index < this->inputs_.size());
        this->outputs_.at(index) = output;
    }

    size_t Layer::input_size() const { return this->inputs_.size(); }

    size_t Layer::output_size() const { return this->outputs_.size(); }

    tensor::Tensor& Layer::get_input(int32_t index)
    {
        CHECK(index >= 0);
        CHECK(index < this->inputs_.size());
        return this->inputs_.at(index);
    }

    const tensor::Tensor& Layer::get_input(int32_t index) const
    {
        CHECK(index >= 0);
        CHECK(index < this->inputs_.size());
        return this->inputs_.at(index);
    }

    tensor::Tensor& Layer::get_output(int32_t index)
    {
        CHECK(index >= 0);
        CHECK(index < this->outputs_.size());
        return this->outputs_.at(index);
    }

    const tensor::Tensor& Layer::get_output(int32_t index) const
    {
        CHECK(index >= 0);
        CHECK(index < this->outputs_.size());
        return this->outputs_.at(index);
    }

    base::Status Layer::check() const
    {
        return base::error::FunctionNotImplement("The check function is not implemented for this layer.");
    }

    base::Status Layer::check_tensor
    (
        const tensor::Tensor& tensor,
        base::DeviceType device_type,
        base::DataType dtype
    ) const
    {
        if (tensor.is_empty())
        {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type)
        {
            return base::error::InvalidArgument("The tensor parameter is not on the same device as the layer.");
        }
        if (tensor.dtype() != dtype)
        {
            return base::error::InvalidArgument("The tensor parameter is not of the same data type as the layer.");
        }
        return base::error::Success();
    }

    base::Status Layer::check_tensor_with_dim
    (
        const tensor::Tensor& tensor,
        base::DeviceType device_type,
        base::DataType dtype,
        int32_t expected_dims,
        ...
    ) const
    {
        if (tensor.is_empty())
        {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type)
        {
            return base::error::InvalidArgument("The tensor parameter is not on the same device as the layer.");
        }
        if (tensor.dtype() != dtype)
        {
            return base::error::InvalidArgument("The tensor parameter is not of the same data type as the layer.");
        }
        int32_t dims = tensor.dims_size();
        if (dims != expected_dims)
        {
            return base::error::InvalidArgument("The tensor parameter has an unexpected number of dimensions.");
        }

        std::va_list args;
        va_start(args, dtype);
        
        for (int32_t i = 0; i < dims; i++)
        {
            int32_t dim = va_arg(args, int32_t);
            if (tensor.get_dim(i) != dim)
            {
                return base::error::InvalidArgument("The tensor parameter has an unexpected dimension size at index " + std::to_string(i) + ".");
            }
        }
        va_end(args);
        return base::error::Success();
    }

    base::Status Layer::forward()
    {
        return base::error::FunctionNotImplement("The forward function is not implemented for this layer.");
    }

    base::Status Layer::forward
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& output1
    )
    {
        this->set_input(0, input1);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output1
    )
    {
        this->set_input(0, input1);
        this->set_input(1, input2);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& input3,
        const tensor::Tensor& output1
    )
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& input3,
        const tensor::Tensor& input4,
        const tensor::Tensor& output1
    )
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward
    (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& input3,
        const tensor::Tensor& input4,
        const tensor::Tensor& input5,
        const tensor::Tensor& output1
    )
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);

        this->set_output(0, output1);
        return this->forward();
    }

    void Layer::reset_input_size(size_t size)
    {
        this->inputs_.resize(size);
    }
    
    void Layer::reset_output_size(size_t size)
    {
        this->outputs_.resize(size);
    }

    void Layer::to_cuda()
    {
        for (auto& input : this->inputs_)
        {
            if (input.is_empty() == false)
            {
                input.to_cuda(this->cuda_config_ ? this->cuda_config_->stream : nullptr);
            }
        }

        for (auto& output : this->outputs_)
        {
            if (output.is_empty() == false)
            {
                output.to_cuda(this->cuda_config_ ? this->cuda_config_->stream : nullptr);
            }
        }
    }

    void Layer::to_cpu()
    {
        for (auto& input : this->inputs_)
        {
            if (input.is_empty() == false)
            {
                input.to_cpu();
            }
        }

        for (auto& output : this->outputs_)
        {
            if (output.is_empty() == false)
            {
                output.to_cpu();
            }
        }
    }

    void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> cuda_config)
    {
        if (cuda_config == nullptr)
        {
            LOG(WARNING) << "The cuda_config parameter is nullptr. The layer will not use stream.";
            return;
        }
        this->cuda_config_ = cuda_config;
    }

    std::shared_ptr<kernel::CudaConfig> Layer::get_cuda_config() const
    {
        return this->cuda_config_;
    }
    // End of Layer

    // LayerParam
    LayerParam::LayerParam
    (
        base::DeviceType device_type,
        LayerType layer_type,
        bool is_quant_layer,
        std::string layer_name
    )
    : Layer(device_type, layer_type, layer_name), is_quant_layer_(is_quant_layer)
    {}

    base::Status LayerParam::set_weight(int32_t index, const tensor::Tensor& weight)
    {
        CHECK(index >= 0);
        CHECK(index < this->weights_.size());
        // NOTE: 为什么必须是FP32的？
        CHECK(weight.dtype() == base::DataType::DataTypeFp32);
        if (weight.is_empty() == false)
        {
            CHECK(weight.device_type() == this->device_type());
        }
        this->weights_.at(index) = weight;
        return base::error::Success();
    }

    base::Status LayerParam::set_weight
    (
        int32_t index,
        const std::vector<int32_t>& dims,
        const void* weight_ptr,
        base::DeviceType device_type
    )
    {
        CHECK(index >= 0);
        CHECK(index < this->weights_.size());
        CHECK(weight_ptr != nullptr);

        size_t size = tensor::reduce_dimensions(dims.begin(), dims.end(), 1);
        // NOTE:属于外部引用的数据
        std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);

        // NOTE: 先初始化buffer，在挪动设备，因为是外部引用，改一下设备位置即可
        CHECK(device_type != base::DeviceType::DeviceUnknown);
        buffer->set_device_type(device_type);
        

        if (this->is_quant_layer_ == false)
        {
            // 没搞从buffer初始化，所以先构造tensor，然后把buffer挪到tensor上
            tensor::Tensor weight(base::DataType::DataTypeFp32, dims);
            weight.set_device_type(device_type);

            CHECK(weight.assign(buffer));
            this->weights_.at(index) = weight;
        }
        else
        {
            // 使用quant layer
            tensor::Tensor weight(base::DataType::DataTypeInt8, dims);
            weight.set_device_type(device_type);

            CHECK(weight.assign(buffer));
            this->weights_.at(index) = weight;

            const int32_t weight_size = static_cast<int32_t>(weight.size());
            CHECK(weight_size % this->group_size_ == 0);

            int32_t scale_nums = weight_size / this->group_size_;
            // 量化比例可能是存储在weight_ptr之后的
            this->scales_ = tensor::Tensor(base::DataType::DataTypeFp32, scale_nums, false, nullptr, reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size));
            this->scales_.set_device_type(device_type);
        }

        return base::error::Success();
    }

    tensor::Tensor& LayerParam::get_weight(int32_t index)
    {
        CHECK(index >= 0);
        CHECK(index < this->weights_.size());
        return this->weights_.at(index);
    }

    const tensor::Tensor& LayerParam::get_weight(int32_t index) const
    {
        CHECK(index >= 0);
        CHECK(index < this->weights_.size());
        return this->weights_.at(index);
    }

    size_t LayerParam::weight_size() const { return this->weights_.size(); }

    void LayerParam::reset_weight_size(size_t size)
    {
        this->weights_.resize(size);
    }

    int32_t LayerParam::get_scales() const
    {
        CHECK(this->scales_.is_empty() == false);
        return static_cast<int32_t>(this->scales_.size());
    }

    void LayerParam::set_scales(const tensor::Tensor& scales)
    {
        CHECK(this->is_quant_layer_);
        CHECK(scales.is_empty() == false);
        this->scales_ = scales;
    }

    void LayerParam::set_group_size(int32_t group_size)
    {
        CHECK(group_size > 0);
        this->group_size_ = group_size;
    }

    void LayerParam::to_cuda()
    {
        Layer::to_cuda();
        for (auto& weight : this->weights_)
        {
            if (weight.is_empty() == false)
            {
                weight.to_cuda(this->cuda_config_ ? this->cuda_config_->stream : nullptr);
            }
        }
        if (this->is_quant_layer_)
        {
            if (this->scales_.is_empty() == false)
            {
                this->scales_.to_cuda(this->cuda_config_ ? this->cuda_config_->stream : nullptr);
            }
        }
    }

    void LayerParam::to_cpu()
    {
        Layer::to_cpu();
        for (auto& weight : this->weights_)
        {
            if (weight.is_empty() == false)
            {
                weight.to_cpu();
            }
        }
        if (this->is_quant_layer_)
        {
            if (this->scales_.is_empty() == false)
            {
                this->scales_.to_cpu();
            }
        }
    }
    









} // namespace op