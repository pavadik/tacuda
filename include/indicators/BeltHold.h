#ifndef BELTHOLD_H
#define BELTHOLD_H

#include "Indicator.h"

class BeltHold : public Indicator {
public:
    BeltHold() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

