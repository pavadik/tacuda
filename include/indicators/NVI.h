#ifndef NVI_H
#define NVI_H

#include "Indicator.h"

namespace tacuda {
class NVI : public Indicator {
public:
    NVI() = default;
    void calculate(const float* close, const float* volume, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
