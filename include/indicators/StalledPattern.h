#ifndef STALLEDPATTERN_H
#define STALLEDPATTERN_H

#include "Indicator.h"

namespace tacuda {
class StalledPattern : public Indicator {
public:
    StalledPattern() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif

