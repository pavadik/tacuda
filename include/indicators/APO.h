#ifndef APO_H
#define APO_H

#include "Indicator.h"

class APO : public Indicator {
public:
    APO(int fastPeriod, int slowPeriod);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int fastPeriod;
    int slowPeriod;
};

#endif
