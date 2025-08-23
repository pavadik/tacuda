#ifndef GAPSIDE_SIDE_WHITE_H
#define GAPSIDE_SIDE_WHITE_H

#include "Indicator.h"

class GapSideSideWhite : public Indicator {
public:
    GapSideSideWhite() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output,
                   int size) noexcept(false) override;
};

#endif

