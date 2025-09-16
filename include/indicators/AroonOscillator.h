#ifndef AROONOSCILLATOR_H
#define AROONOSCILLATOR_H

#include "Indicator.h"

namespace tacuda {
class AroonOscillator : public Indicator {
public:
    explicit AroonOscillator(int period);
    void calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
