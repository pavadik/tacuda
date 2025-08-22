#ifndef HT_DCPHASE_H
#define HT_DCPHASE_H

#include "Indicator.h"

class HT_DCPHASE : public Indicator {
public:
    HT_DCPHASE() = default;
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif
