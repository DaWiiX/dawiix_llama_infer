#include "op/kernels/cpu/matmul_kernel.h"
#include "op/kernels/kernels_interface.h"
#include "base/base.h"

namespace kernel
{
    void matmul_kernel_cpu(
        const tensor::Tensor &input,
        const tensor::Tensor &weight,
        const tensor::Tensor &output,
        float scale,
        const CudaConfig *config)
    {
        UNUSED(config);

        const float *input_ptr = input.ptr<float>();
        const float *weight_ptr = weight.ptr<float>();
        const float *output_ptr = output.ptr<float>();

        int32_t in_dim0 = 1;
        int32_t in_dim1 = 1;

        // actually input is 1D tensor
        if (input.dims_size() == 2)
        {
            in_dim0 = input.get_dim(0);
            in_dim1 = input.get_dim(1);
        }
        else if (input.dims_size() == 1)
        {
            in_dim0 = input.get_dim(0);
        }
        else
            LOG(FATAL) << "Input tensor must be 1D or 2D.";

        const int32_t weight_dim0 = weight.get_dim(0);
        const int32_t weight_dim1 = weight.get_dim(1);

        arma::fmat input_mat(const_cast<float *>(input_ptr), in_dim1, in_dim0, false, true);
        arma::fmat weight_mat(const_cast<float *>(weight_ptr), weight_dim1, weight_dim0, false, true);
        arma::fmat output_mat(const_cast<float *>(output_ptr), in_dim1, weight_dim1, false, true);

        output_mat = (input_mat * weight_mat) * scale;
    }
} // namespace kernel