#ifndef SPINNINGTOP_H
#define SPINNINGTOP_H

#include "Indicator.h"

namespace tacuda {
class SpinningTop : public Indicator {
public:
    SpinningTop() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif

