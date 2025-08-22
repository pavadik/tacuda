#ifndef ADVANCEBLOCK_H
#define ADVANCEBLOCK_H

#include "Indicator.h"

class AdvanceBlock : public Indicator {
public:
    AdvanceBlock() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

