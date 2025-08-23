#ifndef HT_SINE_H
#define HT_SINE_H

#include "Indicator.h"

class HT_SINE : public Indicator {
public:
    HT_SINE() = default;
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

#endif
