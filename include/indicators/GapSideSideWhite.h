#ifndef GAPSIDE_SIDE_WHITE_H
#define GAPSIDE_SIDE_WHITE_H

#include "Indicator.h"

namespace tacuda {
class GapSideSideWhite : public Indicator {
public:
    GapSideSideWhite() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif

