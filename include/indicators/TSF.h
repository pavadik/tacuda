#ifndef TSF_H
#define TSF_H

#include "Indicator.h"

class TSF : public Indicator {
public:
    explicit TSF(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
