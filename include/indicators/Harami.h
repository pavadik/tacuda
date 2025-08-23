#ifndef HARAMI_H
#define HARAMI_H

#include "Indicator.h"

class Harami : public Indicator {
public:
    Harami() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output,
                   int size) noexcept(false) override;
};

#endif

