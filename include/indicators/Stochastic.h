#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "Indicator.h"

namespace tacuda {
class Stochastic : public Indicator {
public:
    Stochastic(int kPeriod, int dPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int kPeriod;
    int dPeriod;
};

} // namespace tacuda

#endif
