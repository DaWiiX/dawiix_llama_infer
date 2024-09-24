#ifndef INCLUDE_OP_KERNELS_CUDA_ARGMAX_KERNEL_H_
#define INCLUDE_OP_KERNELS_CUDA_ARGMAX_KERNEL_H_

namespace op
{
    size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
} // namespace op


#endif // INCLUDE_OP_KERNELS_CUDA_ARGMAX_KERNEL_H_