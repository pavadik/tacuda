#ifndef ULTOSC_H
#define ULTOSC_H

#include "Indicator.h"

class ULTOSC : public Indicator {
public:
    ULTOSC(int shortPeriod, int mediumPeriod, int longPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int shortPeriod;
    int mediumPeriod;
    int longPeriod;
};

#endif
