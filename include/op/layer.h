#ifndef INCLUDE_OP_LAYER_H_
#define INCLUDE_OP_LAYER_H_
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
            std::string layer_name_;
            LayerType layer_type_ = LayerType::LayerUnknown;
            base::DataType dtype_ = base::DataType::DataTypeUnknown;
            base::DeviceType device_type_ = base::DeviceType::DeviceUnknown;

        public:
            explicit BaseLayer
            (
                base::DeviceType device_type,
                LayerType layer_type,
                base::DataType dtype,
                std::string layer_name = ""
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

            virtual void set_output(int32_t index, const tensor::Tensor& output) = 0;

            virtual size_t input_size() const = 0;

            virtual size_t output_size() const = 0;

            virtual tensor::Tensor& get_input(int32_t index) = 0;

            virtual const tensor::Tensor& get_input(int32_t index) const = 0;

            virtual tensor::Tensor& get_output(int32_t index) = 0;

            virtual const tensor::Tensor& get_output(int32_t index) const = 0;

            virtual base::Status check() const = 0;
            
            virtual base::Status forward() = 0;

            virtual base::Status forward
            (
                const tensor::Tensor& input1,
                const tensor::Tensor& output1
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &output1
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &output1
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &output1
            ) = 0;

            virtual base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &input5,
                const tensor::Tensor &output1
            ) = 0;
    };


    class Layer : public BaseLayer 
    {
        protected:
            std::vector<tensor::Tensor> inputs_;
            std::vector<tensor::Tensor> outputs_;
            std::shared_ptr<kernel::CudaConfig> cuda_config_ = nullptr;

        public:
            explicit Layer
            (
                base::DeviceType device_type,
                LayerType layer_type,
                std::string layer_name = ""
            );

            base::Status init() override;

            void set_input(int32_t index, const tensor::Tensor& input) override;

            void set_output(int32_t index, const tensor::Tensor& output) override;

            size_t input_size() const override;

            size_t output_size() const override;

            tensor::Tensor& get_input(int32_t index) override;

            const tensor::Tensor& get_input(int32_t index) const override;

            tensor::Tensor& get_output(int32_t index) override;

            const tensor::Tensor& get_output(int32_t index) const override;

            base::Status check() const override;

            base::Status check_tensor
            (
                const tensor::Tensor& tensor,
                base::DeviceType device_type,
                base::DataType dtype
            ) const;

            base::Status check_tensor_with_dim
            (
                const tensor::Tensor& tensor,
                base::DeviceType device_type,
                base::DataType dtype,
                int32_t expected_dims,
                ...
            ) const;

            base::Status forward() override;

            base::Status forward
            (
                const tensor::Tensor& input1,
                const tensor::Tensor& output1
            ) override;

            base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &output1
            ) override;

            base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &output1
            ) override;

            base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &output1
            ) override;

            base::Status forward
            (
                const tensor::Tensor &input1,
                const tensor::Tensor &input2,
                const tensor::Tensor &input3,
                const tensor::Tensor &input4,
                const tensor::Tensor &input5,
                const tensor::Tensor &output1
            ) override;

            void reset_input_size(size_t size);

            void reset_output_size(size_t size);

            virtual void to_cuda();

            virtual void to_cpu();

            void set_cuda_config(std::shared_ptr<kernel::CudaConfig> cuda_config);

            std::shared_ptr<kernel::CudaConfig> get_cuda_config() const;
    };

    class LayerParam : public Layer
    {
        protected:
            int32_t group_size_ = 0;
            bool is_quant_layer_ = false;
            tensor::Tensor scales_;
            std::vector<tensor::Tensor> weights_;
        
        public:
            explicit LayerParam
            (
                base::DeviceType device_type,
                LayerType layer_type,
                bool is_quant_layer,
                std::string layer_name = ""
            );

            base::Status set_weight(int32_t index, const tensor::Tensor& weight) override;

            base::Status set_weight
            (
                int32_t index,
                const std::vector<int32_t>& dims,
                const void* weight_ptr,
                base::DeviceType device_type = base::DeviceType::DeviceUnknown
            );

            tensor::Tensor& get_weight(int32_t index);

            const tensor::Tensor& get_weight(int32_t index) const;

            size_t weight_size() const;

            void reset_weight_size(size_t size);

            int32_t get_scales() const;

            void set_scales(const tensor::Tensor& scales);

            void set_group_size(int32_t group_size);

            void to_cuda() override;

            void to_cpu() override;
    };
}
#endif // _INCLUDE_OP_LAYER_H_