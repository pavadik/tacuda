#ifndef MIDPOINT_H
#define MIDPOINT_H

#include "Indicator.h"

class MIDPOINT : public Indicator {
public:
    explicit MIDPOINT(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
