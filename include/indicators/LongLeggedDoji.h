#ifndef LONGLEGGEDDOJI_H
#define LONGLEGGEDDOJI_H

#include "Indicator.h"

class LongLeggedDoji : public Indicator {
public:
    LongLeggedDoji() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output,
                   int size) noexcept(false) override;
};

#endif

