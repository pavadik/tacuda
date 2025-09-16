#ifndef BBANDS_H
#define BBANDS_H

#include "Indicator.h"

namespace tacuda {
class BBANDS : public Indicator {
public:
    BBANDS(int period, float upperMultiplier, float lowerMultiplier);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
    float upperMultiplier;
    float lowerMultiplier;
};

} // namespace tacuda

#endif
