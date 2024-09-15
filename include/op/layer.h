#ifndef __INCLUDE_OP_LAYER_H_
#define __INCLUDE_OP_LAYER_H_
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op
{
    enum class LayerType : uint8_t
    {
        LayerUnknown = 0,
        LayerLinear = 1,
        LayerEncode = 2,
        LayerEmbedding = 3,
        LayerRMSNorm = 4,
        LayerMatmul = 5,
        LayerRoPe = 6,
        LayerMHA = 7,
        LayerSoftmax = 8,
        LayerAdd = 9,
        LayerSwiGLU = 10,
    };

    class BaseLayer
    {
        protected:
            std:string layer_name_;
            LayerType layer_type_ = LayerType::layerUnknown;
            base::DataType dtype_ = base::DataType::DataTypeUnknown;
            base::DeviceType device_type_ = base::DeviceType::DeviceUnknown;

        public:
            explicit BaseLayer
            (
                base::DeviceType device_type,
                base::LayerType layer_type,
                base::DataType dtype,
                const std::string& layer_name = ""
            );

            base::DataType dtype() const;

            base::DeviceType device_type() const;
            void set_device_type(base::DeviceType device_type);

            LayerType layer_type() const;

            const std::string& get_layer_name() const;
            void set_layer_name(const std::string& layer_name);

            virtual base::Status init() = 0;

            virtual void set_input(int32_t index, const tensor::Tensor& input) = 0;

            virtual base::Status set_weight(int32_t index, const tensor::Tensor& weight);

            virtual base::Status set_weight
            (
                int32_t index,
                const std::vector<int32_t>& dims,
                const void* weight_ptr,
                base::DeviceType device_type = base::DeviceType::DeviceUnknown
            );

            virtual void set_output(int32_t index, const tensor::Tensor& output);

            virtual size_t input_size() const = 0;

            virtual size_t output_size() const = 0;

            virtual tensor::Tensor& get_input(int32_t index) = 0;
            virtual const tensor::Tensor& get_input(int32_t index) = 0;

            virtual tensor::Tensor& get_output(int32_t index) = 0;
            virtual const tensor::Tensor& get_output(int32_t index) = 0;

            virtual base::Status check() const = 0;

            virtual base::Status forward
            (
                const tensor::Tensor& input1,
                const tensor::Tensor& output1
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &output1, 
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &output1, 
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &output1, 
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &input5,
                const tensor::Tensor &output1, 
            ) = 0;
    }





}



#endif // __INCLUDE_OP_LAYER_H_