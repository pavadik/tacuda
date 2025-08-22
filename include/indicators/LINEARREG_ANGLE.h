#ifndef LINEARREG_ANGLE_H
#define LINEARREG_ANGLE_H

#include "Indicator.h"

class LINEARREG_ANGLE : public Indicator {
public:
    explicit LINEARREG_ANGLE(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
