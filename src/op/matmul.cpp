#include "op/matmul.h"
#include "op/kernels/kernels_interface.h"

namespace op
{
    MatmulLayer::MatmulLayer
    (
        base::DeviceType device_type,
        int32_t dim0,
        int32_t dim1,
        bool is_quant_layer
    )
    : LayerParam(device_type, LayerType::LayerMatmul, is_quant_layer, "Matmul"), dim0_(dim0), dim1_(dim1)
    {
        this->reset_input_size(1);
        this->reset_weight_size(1);
        this->reset_output_size(1);
    }

    base::Status MatmulLayer::check() const
    {
        auto status = this->check_tensor_with_dim(
            this->get_input(0),
            this->device_type_,
            this->dtype_,
            1,
            this->dim1_
        );
        if (!status)
        {
            LOG(ERROR) << "Input tensor must be a vector with dim " << this->dim1_ << "in the first input of Matmul layer.";
            return status;
        }

        status = this->check_tensor_with_dim(
            this->get_output(0),
            this->device_type_,
            this->dtype_,
            1,
            this->dim0_
        );
        if (!status)
        {
            LOG(ERROR) << "Output tensor must be a vector with dim " << this->dim0_ << "in the first output of Matmul layer.";
            return status;
        }

        if (this->is_quant_layer_ == false)
        {
            status = this->check_tensor_with_dim(
                this->get_weight(0),
                this->device_type_,
                this->dtype_,
                2,
                this->dim0_,
                this->dim1_
            );
            if (!status)
            {
                LOG(ERROR) << "Weight tensor must be a matrix with dim " << this->dim0_ << "x" << this->dim1_ << "in the first weight of Matmul layer.";
                return status;
            };
        }
        else
        {
            status = this->check_tensor_with_dim(
                this->get_weight(0),
                this->device_type_,
                base::DataType::DataTypeInt8,
                2,
                this->dim0_,
                this->dim1_
            );
            if (!status)
            {
                LOG(ERROR) << "Weight tensor must be a matrix with dim " << this->dim0_ << "x" << this->dim1_ << "in the first weight of quantized Matmul layer.";
                return status;
            };

            status = check_tensor_with_dim(
                this->scales_, 
                this->device_type_, 
                base::DataType::DataTypeFp32, 
                1,
                this->scales_.size()
            );
            if (!status)
            {
                LOG(ERROR) << "The scale tensor error in the matmul layer.";
                return status;
            }
        }

        return base::error::Success();
    }

    base::Status MatmulLayer::forward()
    {
        auto status = check();
        if (!status) {
        return status;
        }
        if (device_type_ == base::DeviceType::DeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
        }
        if (is_quant_layer_) {
            kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                        group_size_, scales_,
                                                        cuda_config_ ? cuda_config_.get() : nullptr);
        } else {
            kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                                    cuda_config_ ? cuda_config_.get() : nullptr);
        }

        return base::error::Success();
    }
}