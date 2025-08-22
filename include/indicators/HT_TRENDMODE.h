#ifndef HT_TRENDMODE_H
#define HT_TRENDMODE_H

#include "Indicator.h"

class HT_TRENDMODE : public Indicator {
public:
    HT_TRENDMODE() = default;
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif
