#ifndef EMA_H
#define EMA_H

#include "Indicator.h"

class EMA : public Indicator {
public:
    explicit EMA(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
