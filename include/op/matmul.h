#ifndef INCLUDE_OP_MATMUL_H_
#define INCLUDE_OP_MATMUL_H_

#include <base/cuda_config.h>
#include "layer.h"

namespace op
{
    class MatmulLayer : public LayerParam
    {
        private:
            int32_t dim0_ = 0;
            int32_t dim1_ = 0;

        public:
            explicit MatmulLayer
            (
                base::DeviceType device_type,
                int32_t dim0,
                int32_t dim1,
                bool is_quant_layer = false
            );

            base::Status check() const override;

            base::Status forward() override;
    };
} // namespace op

#endif // INCLUDE_OP_MATMUL_H_