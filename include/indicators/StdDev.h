#ifndef STDDEV_H
#define STDDEV_H

#include "Indicator.h"

class StdDev : public Indicator {
public:
    explicit StdDev(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
