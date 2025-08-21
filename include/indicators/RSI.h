#ifndef RSI_H
#define RSI_H

#include "Indicator.h"

class RSI : public Indicator {
public:
    explicit RSI(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
