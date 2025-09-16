#ifndef BEARISHENGULFING_H
#define BEARISHENGULFING_H

#include "Indicator.h"

namespace tacuda {
class BearishEngulfing : public Indicator {
public:
    BearishEngulfing() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
