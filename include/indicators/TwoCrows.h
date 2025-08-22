#ifndef TWOCROWS_H
#define TWOCROWS_H

#include "Indicator.h"

class TwoCrows : public Indicator {
public:
    TwoCrows() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

