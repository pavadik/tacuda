#ifndef VAR_H
#define VAR_H

#include "Indicator.h"

class VAR : public Indicator {
public:
    explicit VAR(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
