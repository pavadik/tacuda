#ifndef ATR_H
#define ATR_H

#include "Indicator.h"

namespace tacuda {
class ATR : public Indicator {
public:
    explicit ATR(int period, float initial = 0.0f);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
    float initial;
};

} // namespace tacuda

#endif
