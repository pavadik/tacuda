#ifndef HT_TRENDMODE_H
#define HT_TRENDMODE_H

#include "Indicator.h"

namespace tacuda {
class HT_TRENDMODE : public Indicator {
public:
    HT_TRENDMODE() = default;
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
