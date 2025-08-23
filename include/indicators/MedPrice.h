#ifndef MEDPRICE_H
#define MEDPRICE_H

#include "Indicator.h"

class MedPrice : public Indicator {
public:
    MedPrice() = default;
    void calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

#endif
