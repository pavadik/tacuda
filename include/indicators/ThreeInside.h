#ifndef THREEINSIDE_H
#define THREEINSIDE_H

#include "Indicator.h"

class ThreeInside : public Indicator {
public:
    ThreeInside() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

