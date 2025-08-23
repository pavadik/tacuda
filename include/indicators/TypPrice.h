#ifndef TYPPRICE_H
#define TYPPRICE_H

#include "Indicator.h"

class TypPrice : public Indicator {
public:
    TypPrice() = default;
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

#endif
