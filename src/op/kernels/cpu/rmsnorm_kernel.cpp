#include "op/kernels/cpu/rmsnorm_kernel.h"

namespace kernel
{
    void rmsnorm_kernel_cpu
    (
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        void* stream
    )
    {
        UNUSED(stream);

        const float* input_ptr = input.ptr<float>();
        const float* weight_ptr = weight.ptr<float>();
        const float* output_ptr = output.ptr<float>();

        const int32_t dim = static_cast<int32_t>(input.size());
        // 3rd facotr is copy_aux_mem, false means we use the original mem
        // 4th factor is strict , true means we can only modify the data, the size is forbiden to change
        arma::fvec input_tensor(const_cast<float*>(input_ptr), dim, false, true);
        arma::fvec weight_tensor(const_cast<float*>(weight_ptr), dim, false, true);
        arma::fvec output_tensor(const_cast<float*>(output_ptr), dim, false, true);
        
        const float epsilon = 1e-5f;
        const float mean = arma::as_scalar(arma::mean(arma::pow(input_tensor, 2)) + epsilon);
        const float rsqrt = 1.f / std::sqrt(mean);

        // %	element-wise multiplication of two objects (Schur product)
        output_tensor = weight_tensor % (input_tensor * rsqrt);
    }
}