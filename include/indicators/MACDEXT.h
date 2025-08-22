#ifndef MACDEXT_H
#define MACDEXT_H

#include "Indicator.h"
#include "MA.h"

class MACDEXT : public Indicator {
public:
    MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, MAType type);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int fastPeriod;
    int slowPeriod;
    int signalPeriod;
    MAType type;
};

#endif
