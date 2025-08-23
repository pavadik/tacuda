#ifndef MACD_H
#define MACD_H

#include "Indicator.h"

class MACD : public Indicator {
public:
    MACD(int fastPeriod, int slowPeriod);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int fastPeriod;
    int slowPeriod;
};

#endif
