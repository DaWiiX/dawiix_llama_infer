#ifndef INCLUDE_SAMPLER_SAMPLER_H_
#define INCLUDE_SAMPLER_SAMPLER_H_

#include <cstddef>
#include <cstdint>

namespace sampler
{
    class Sampler
    {
    public:
        explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

        virtual size_t sample(const float *logits, size_t size, void *stream = nullptr) = 0;

    protected:
        base::DeviceType device_type_;
    };
} // namespace sampler

#endif // INCLUDE_SAMPLER_SAMPLER_H_