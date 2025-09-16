#ifndef LINEARREG_SLOPE_H
#define LINEARREG_SLOPE_H

#include "Indicator.h"

namespace tacuda {
class LINEARREG_SLOPE : public Indicator {
public:
    explicit LINEARREG_SLOPE(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
