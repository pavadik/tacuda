#ifndef BBANDS_H
#define BBANDS_H

#include "Indicator.h"

class BBANDS : public Indicator {
public:
    BBANDS(int period, float upperMultiplier, float lowerMultiplier);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
    float upperMultiplier;
    float lowerMultiplier;
};

#endif
