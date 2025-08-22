#ifndef AROON_H
#define AROON_H

#include "Indicator.h"

class Aroon : public Indicator {
public:
    Aroon(int upPeriod, int downPeriod);
    void calculate(const float* high, const float* low, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int upPeriod;
    int downPeriod;
};

#endif
