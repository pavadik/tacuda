#ifndef TEMA_H
#define TEMA_H

#include "Indicator.h"

class TEMA : public Indicator {
public:
    explicit TEMA(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
