#ifndef THREEWHITESOLDIERS_H
#define THREEWHITESOLDIERS_H

#include "Indicator.h"

class ThreeWhiteSoldiers : public Indicator {
public:
    ThreeWhiteSoldiers() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif

