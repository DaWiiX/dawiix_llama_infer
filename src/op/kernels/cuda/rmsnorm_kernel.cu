#include <cub/block/block_reduce.cuh>
#include "op/kernels/cuda/rmsnorm_kernel.cuh"

namespace kernel
{
    // cu function donot support reference param
    template<int32_t BLOCKDIM>
    static __global__ void row_rmsnorm_fp32
    (
        float* input,
        float* weight,
        float* output,
        int size,
        float eps
    )
    {
        const int tid = threadIdx.x;
        constexpr int packnum = 4;
        const int pack_num = size / packnum;
        const int pack_all = pack_num * packnum;

        float sum = 0.0f;
        float4* input_pack = reinterpret_cast<float4*>(input);
        for (int i = tid; i < pack_num; i += blockDim.x)
        {
            float4 input_float4 = *(input_pack + i);
            sum += input_float4.x * input_float4.x + input_float4.y * input_float4.y + input_float4.z * input_float4.z + input_float4.w * input_float4.w;
        }
        for (int i = pack_all + tid; i < size; i += blockDim.x)
        {
            sum += input[i] * input[i];
        }

        using BlockReduce = cub::BlockReduce<float, BLOCKDIM>;
        // TempStorage is a struct so we need to use typename to refer to it
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float shared_val;

        sum = BlockReduce(temp_storage).Sum(sum);
        if (tid == 0) shared_val = sum;
        __syncthreads();

        sum = shared_val;
        const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

        float4* output_pack = reinterpret_cast<float4*>(output);
        float4* weight_pack = reinterpret_cast<float4*>(weight);
        for (int i = tid; i < pack_num; i += blockDim.x)
        {
            float4 input_float4 = *(input_pack + i);
            float4 weight_float4 = *(weight_pack + i);

            *(output_pack+i) = make_float4(
                input_float4.x * weight_float4.x * scale,
                input_float4.y * weight_float4.y * scale,
                input_float4.z * weight_float4.z * scale,
                input_float4.w * weight_float4.w * scale
            );
        }

        for (int i = pack_all + tid; i < size; i += blockDim.x)
        {
            output[i] = input[i] * weight[i] * scale;
        }
    }

    void rmsnorm_kernel_cu
    (
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        void* stream
    )
    {
        float* input_ptr = const_cast<float*>(input.ptr<float>());
        float* weight_ptr = const_cast<float*>(weight.ptr<float>());
        float* output_ptr = const_cast<float*>(output.ptr<float>());

        const int32_t dim = static_cast<int32_t>(input.size());
        const float eps = 1e-5f;

        constexpr int32_t threads_num = 128;
        dim3 GridSize(1, 1, 1);
        dim3 BlockSize(threads_num, 1, 1);
        if (stream)
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            row_rmsnorm_fp32<threads_num><<<GridSize, BlockSize, 0, stream_>>>(input_ptr, weight_ptr, output_ptr, dim, eps);
        }
        else
        {
            row_rmsnorm_fp32<threads_num><<<GridSize, BlockSize>>>(input_ptr, weight_ptr, output_ptr, dim, eps);
        }
    }
}