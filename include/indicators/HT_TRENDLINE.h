#ifndef HT_TRENDLINE_H
#define HT_TRENDLINE_H

#include "Indicator.h"

class HT_TRENDLINE : public Indicator {
public:
    HT_TRENDLINE() = default;
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif
