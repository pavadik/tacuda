#ifndef LINEARREG_INTERCEPT_H
#define LINEARREG_INTERCEPT_H

#include "Indicator.h"

class LINEARREG_INTERCEPT : public Indicator {
public:
    explicit LINEARREG_INTERCEPT(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
