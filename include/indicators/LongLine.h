#ifndef LONGLINE_H
#define LONGLINE_H

#include "Indicator.h"

namespace tacuda {
class LongLine : public Indicator {
public:
    LongLine() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif

