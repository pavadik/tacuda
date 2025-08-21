#ifndef CCI_H
#define CCI_H

#include "Indicator.h"

class CCI : public Indicator {
public:
    explicit CCI(int period);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
