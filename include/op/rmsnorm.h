#ifndef INCLUDE_OP_RMSNORM_H_
#define INCLUDE_OP_RMSNORM_H_

#include "layer.h"

namespace op 
{
    class RMSNormLayer : public LayerParam
    {
        private:
            int32_t dim_;
        
        public:
            explicit RMSNormLayer(base::DeviceType device_type, int32_t dim);

            base::Status check() const override;

            base::Status forward() override;
    };
}  // namespace op

#endif  // INCLUDE_OP_RMSNORM_H_