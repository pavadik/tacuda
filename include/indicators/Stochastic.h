#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "Indicator.h"

class Stochastic : public Indicator {
public:
    Stochastic(int kPeriod, int dPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int kPeriod;
    int dPeriod;
};

#endif
