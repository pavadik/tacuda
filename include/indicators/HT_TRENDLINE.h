#ifndef HT_TRENDLINE_H
#define HT_TRENDLINE_H

#include "Indicator.h"

namespace tacuda {
class HT_TRENDLINE : public Indicator {
public:
    HT_TRENDLINE() = default;
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
