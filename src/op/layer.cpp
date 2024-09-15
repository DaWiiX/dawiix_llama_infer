#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>
#include <base/cuda_config.h>
#include "op/layer.h"


namespace op
{
    //
    BaseLayer::BaseLayer
    (
        base::DeviceType device_type,
        LayerType layer_type,
        base::DataType dtype,
        const std::string& layer_name
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

} // namespace op