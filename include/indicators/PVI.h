#ifndef PVI_H
#define PVI_H

#include "Indicator.h"

namespace tacuda {
class PVI : public Indicator {
public:
    PVI() = default;
    void calculate(const float* close, const float* volume, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
