#ifndef DX_H
#define DX_H

#include "Indicator.h"

class DX : public Indicator {
public:
    explicit DX(int period);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
