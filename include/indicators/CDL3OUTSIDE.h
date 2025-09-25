#ifndef CDL3OUTSIDE_H
#define CDL3OUTSIDE_H

#include "Indicator.h"

namespace tacuda {
class CDL3OUTSIDE : public Indicator {
public:
    CDL3OUTSIDE() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
