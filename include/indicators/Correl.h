#ifndef CORREL_H
#define CORREL_H

#include "Indicator.h"

class Correl : public Indicator {
public:
    explicit Correl(int period);
    void calculate(const float* x, const float* y, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
