#ifndef HT_PHASOR_H
#define HT_PHASOR_H

#include "Indicator.h"

namespace tacuda {
class HT_PHASOR : public Indicator {
public:
    HT_PHASOR() = default;
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
