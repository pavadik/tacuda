#ifndef LINEARREG_H
#define LINEARREG_H

#include "Indicator.h"

class LINEARREG : public Indicator {
public:
    explicit LINEARREG(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
