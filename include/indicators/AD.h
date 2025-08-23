#ifndef AD_H
#define AD_H

#include "Indicator.h"

class AD : public Indicator {
public:
    AD() = default;
    void calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

#endif
