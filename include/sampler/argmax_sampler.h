#ifndef INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_
#define INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_

#include <base/base.h>
#include "sampler.h"

namespace sampler
{
    class ArgmaxSampler : public Sampler
    {
    public:
        explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

        size_t sample(const float *logits, size_t size, void *stream) override;
    };
} // namespace sampler

#endif // INCLUDE_SAMPLER_ARGMAX_SAMPLER_H_