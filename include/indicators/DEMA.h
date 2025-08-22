#ifndef DEMA_H
#define DEMA_H

#include "Indicator.h"

class DEMA : public Indicator {
public:
    explicit DEMA(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
