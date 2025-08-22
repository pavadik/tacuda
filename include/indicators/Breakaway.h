#ifndef BREAKAWAY_H
#define BREAKAWAY_H

#include "Indicator.h"

class Breakaway : public Indicator {
public:
    Breakaway() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

