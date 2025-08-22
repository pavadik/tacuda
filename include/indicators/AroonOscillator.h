#ifndef AROONOSCILLATOR_H
#define AROONOSCILLATOR_H

#include "Indicator.h"

class AroonOscillator : public Indicator {
public:
    explicit AroonOscillator(int period);
    void calculate(const float* high, const float* low, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
