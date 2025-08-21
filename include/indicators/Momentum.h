#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "Indicator.h"

class Momentum : public Indicator {
public:
    explicit Momentum(int period);
    void calculate(const float* input, float* output, int size) override;
private:
    int period;
};

#endif
