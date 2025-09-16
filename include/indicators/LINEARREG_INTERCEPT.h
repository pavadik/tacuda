#ifndef LINEARREG_INTERCEPT_H
#define LINEARREG_INTERCEPT_H

#include "Indicator.h"

namespace tacuda {
class LINEARREG_INTERCEPT : public Indicator {
public:
    explicit LINEARREG_INTERCEPT(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
