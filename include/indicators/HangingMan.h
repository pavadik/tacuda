#ifndef HANGINGMAN_H
#define HANGINGMAN_H

#include "Indicator.h"

class HangingMan : public Indicator {
public:
    HangingMan() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false) override;
};

#endif

