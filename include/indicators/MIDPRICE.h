#ifndef MIDPRICE_H
#define MIDPRICE_H

#include "Indicator.h"

class MIDPRICE : public Indicator {
public:
    explicit MIDPRICE(int period);
    void calculate(const float* high, const float* low, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
