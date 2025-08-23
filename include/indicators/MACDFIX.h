#ifndef MACDFIX_H
#define MACDFIX_H

#include "Indicator.h"
#include "MA.h"

class MACDFIX : public Indicator {
public:
    explicit MACDFIX(int signalPeriod);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int signalPeriod;
};

#endif
