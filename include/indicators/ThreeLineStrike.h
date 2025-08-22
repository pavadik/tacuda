#ifndef THREELINESTRIKE_H
#define THREELINESTRIKE_H

#include "Indicator.h"

class ThreeLineStrike : public Indicator {
public:
    ThreeLineStrike() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

