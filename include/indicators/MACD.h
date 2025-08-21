#ifndef MACD_H
#define MACD_H

#include "Indicator.h"

class MACD : public Indicator {
public:
    MACD(int fastPeriod, int slowPeriod, int signalPeriod);
    void calculate(const float* input, float* output, int size) override;
private:
    int fastPeriod;
    int slowPeriod;
    int signalPeriod;
};

#endif
