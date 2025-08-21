#ifndef SMA_H
#define SMA_H

#include "Indicator.h"

class SMA : public Indicator {
public:
    explicit SMA(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
