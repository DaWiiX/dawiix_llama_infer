#include <cuda_runtime_api.h>
#include <armadillo>
#include "op/rmsnorm.h"
#include "op/kernels/kernels_interface.h"

namespace op
{
    RMSNormLayer::RMSNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::LayerRMSNorm, false, "RMSNorm"), dim_(dim)
    {
        this->reset_weight_size(1);
        this->reset_input_size(1);
        this->reset_output_size(1);
    }

    base::Status RMSNormLayer::check() const
    {
        auto status = this->check_tensor_with_dim(
            this->get_input(0),
            this->device_type_,
            this->dtype_,
            1,
            this->dim_
        );
        if (!status)
        {
            LOG(ERROR) << "Input tensor must be a vector with dim " << this->dim_ << "in the RMSNorm layer.";
            return status;
        }

        status = this->check_tensor_with_dim(
            this->get_output(0),
            this->device_type_,
            this->dtype_,
            1,
            this->dim_
        );
        if (!status)
        {
            LOG(ERROR) << "Output tensor must be a vector with dim " << this->dim_ << "in the RMSNorm layer.";
            return status;
        }

        status = this->check_tensor_with_dim(
            this->get_weight(0),
            this->device_type_,
            this->dtype_,
            1,
            this->dim_
        );
        if (!status)
        {
            LOG(ERROR) << "Weight tensor must be a vector with dim " << this->dim_ << "in the RMSNorm layer.";
            return status;
        }

        return base::error::Success();
    }

    base::Status RMSNormLayer::forward()
    {
        auto status = this->check();
        if (!status)
        {
            return status;
        }

        auto input = this->get_input(0);
        auto output = this->get_output(0);
        auto weight = this->get_weight(0);

        kernel::get_rmsnorm_kernel(this->device_type_)
        (
            input,
            weight,
            output,
            this->cuda_config_ ? this->cuda_config_->stream : nullptr
        );
        return base::error::Success();
    }


} // namespace op