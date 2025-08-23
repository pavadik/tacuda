#ifndef ADOSC_H
#define ADOSC_H

#include "Indicator.h"

class ADOSC : public Indicator {
public:
    ADOSC(int shortPeriod, int longPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int shortPeriod;
    int longPeriod;
};

#endif
