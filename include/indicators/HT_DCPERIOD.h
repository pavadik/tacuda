#ifndef HT_DCPERIOD_H
#define HT_DCPERIOD_H

#include "Indicator.h"

class HT_DCPERIOD : public Indicator {
public:
    HT_DCPERIOD() = default;
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif
