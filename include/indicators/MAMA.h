#ifndef MAMA_H
#define MAMA_H

#include "Indicator.h"

class MAMA : public Indicator {
public:
    MAMA(float fastLimit, float slowLimit);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    float fastLimit;
    float slowLimit;
};

#endif
