#ifndef ADXR_H
#define ADXR_H

#include "Indicator.h"

class ADXR : public Indicator {
public:
    explicit ADXR(int period);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
