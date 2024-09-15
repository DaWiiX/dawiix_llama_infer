#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>
#include <base/cuda_config.h>
#include "op/layer.h"


namespace op
{
    Base::BaseLayer
    (
        base::DeviceType device_type,
        LayerType layer_type,
        base::DataType dtype,
        std::string layer_name,
    )
    : device_type_(device_type),
      layer_type_(layer_type),
      dtype_(dtype),
      layer_name_(std::move(layer_name))
    {}

    base::DataType dtype() const { return this->dtype_; }

    base::DeviceType device_type() const { return this->device_type_; }

    LayerType layer_type() const { return this->layer_type_; }

    const std::string &get_layer_name() const { return this->layer_name_; }

    

} // namespace op